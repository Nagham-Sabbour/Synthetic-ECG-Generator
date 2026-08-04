import torch
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm


import numpy as np
import random
import math
import argparse
import os
import datetime

from model import VAE, Discriminator
from utils import load_vae_checkpoint

DATA_ROOT = os.path.expanduser(
    "./data/ptb-xl-preprocessed-train"
)


def finetune_VAE_GAN(vae, discriminator, num_epochs, train_loader, vae_optimizer, discrim_optimizer, beta=1.0, lambda_adv=1.0, device='cpu', checkpoint_dir='checkpoints'):
    '''
    Fine-tune a pretrained VAE's decoder using a GAN-based discriminator.

    Args:
        vae: pretrained instance of model.VAE
        discriminator: instance of model.Discriminator
        num_epochs: number of epochs to fine-tune for
        train_loader: contains the training data
        vae_optimizer: optimizer to use for training the VAE decoder
        discrim_optimizer: optimizer to use for training the GAN discriminator
        beta: weight factor for KL divergence loss term
        lambda_adv: weight factor for adversarial loss term
        device: 'cpu' or 'cuda'
        checkpoint_dir: directory name of where to save the model checkpoints
    '''

    vae.to(device)
    discriminator.to(device)

    # freeze vae encoder during fine tuning
    for param in vae.encoder.parameters():
        param.requires_grad = False

    # for checkpoint saving
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    best_recon_loss = float('inf')
    best_checkpoint_path = None
    
    for epoch in range(num_epochs):

        # put to train mode
        vae.train()
        discriminator.train()

        # initiate variables
        total_generator_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        generator_adversarial_loss_total = 0
        discrim_loss_total = 0

        batch_num = 0

        # train loop
        for signals, labels in train_loader:
            batch_num += 1

            signals = signals.to(device)
            labels = labels.to(device)

            # TRAIN DISCRIMINATOR:

            with torch.no_grad():
                mean, logvar, recon_out = vae(signals, labels) 

            # Discriminator loss
            real_logits = discriminator(signals, labels)
            fake_logits = discriminator(recon_out.detach(), labels)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            discrim_loss = real_loss + fake_loss

            # Backpropagate and optimize
            discrim_optimizer.zero_grad()
            discrim_loss.backward()
            discrim_optimizer.step()

            # Update losses
            discrim_loss_total += discrim_loss.item()

            # TRAIN GENERATOR (DECODER):

            mean, logvar, recon_out = vae(signals, labels) 

            # Compute reconstruction loss
            recon_loss = F.mse_loss(recon_out, signals, reduction='sum')

            # Compute KL divergence loss
            kl_loss = -0.5 * torch.sum(logvar + 1 - mean.pow(2) - logvar.exp())

            # Compute generator adversarial loss
            fake_logits_for_gen = discriminator(recon_out, labels)
            generator_adversarial_loss = F.binary_cross_entropy_with_logits(fake_logits_for_gen, torch.ones_like(fake_logits_for_gen))

            # Overall generator loss (VAE loss + adversarial loss)
            generator_loss = recon_loss + (beta * kl_loss) + (lambda_adv * generator_adversarial_loss)

            # Backpropagate and optimize
            vae_optimizer.zero_grad()
            generator_loss.backward()
            vae_optimizer.step()

            # Update losses
            total_generator_loss += generator_loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
            generator_adversarial_loss_total += generator_adversarial_loss.item()
                

        num_batches = len(train_loader)
        avg_generator_loss = total_generator_loss / num_batches
        avg_recon_loss = recon_loss_total / num_batches
        avg_kl_loss = kl_loss_total / num_batches
        avg_generator_adversarial_loss = generator_adversarial_loss_total / num_batches
        avg_discrim_loss = discrim_loss_total / num_batches

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
            f"Generator Loss: {avg_generator_loss:.3f}  "
            f"Reconstruction Loss: {avg_recon_loss:.3f}  "
            f"KL Divergence Loss: {avg_kl_loss:.3f}  "
            f"Generator Adversarial Loss: {avg_generator_adversarial_loss:.3f}  "
            f"Discriminator Loss: {avg_discrim_loss:.3f}")

        # save best checkpoint by avg train loss
        if avg_recon_loss < best_recon_loss:
            best_recon_loss = avg_recon_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, f"vae_gan_best_{run_timestamp}.pt")
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'discrim_state_dict': discriminator.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'discrim_optimizer_state_dict': discrim_optimizer.state_dict(),
                'epoch': epoch + 1,
                'recon_loss': avg_recon_loss,
                'generator_loss': avg_generator_loss,
                'discrim_loss': avg_discrim_loss,
            }, best_checkpoint_path)

    # save final checkpoint regardless of loss
    final_checkpoint_path = os.path.join(checkpoint_dir, f"vae_gan_final_epoch{num_epochs}_{run_timestamp}.pt")
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'discrim_state_dict': discriminator.state_dict(),
        'vae_optimizer_state_dict': vae_optimizer.state_dict(),
        'discrim_optimizer_state_dict': discrim_optimizer.state_dict(),
        'epoch': epoch + 1,
        'recon_loss': avg_recon_loss,
        'generator_loss': avg_generator_loss,
        'discrim_loss': avg_discrim_loss,
    }, final_checkpoint_path)

    print(f"Saved best checkpoint (recon_loss={best_recon_loss:.3f}) to {best_checkpoint_path}")
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    return best_checkpoint_path, final_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Synthetic ECG Generator (VAE + GAN)")
    parser.add_argument("--trained-vae-filename", type=str)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--discrim-lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32) #Note: must match the original vae training run
    parser.add_argument("--loss-beta", type=float, default=1.0)
    parser.add_argument("--loss-lambda-adv", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=15) #Note: must match the original vae training run
    parser.add_argument("--checkpoint-dir", type=str, default='checkpoints')

    args = parser.parse_args()
    
    # set the arguments
    trained_vae_filename = args.trained_vae_filename
    embedding_dim = args.embedding_dim
    num_classes = args.num_classes
    decoder_lr = args.decoder_lr
    discrim_lr = args.discrim_lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    beta = args.loss_beta
    lambda_adv = args.loss_lambda_adv
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.checkpoint_dir

    # load the preprocessed dataset - TODO
    #train_dataset = 
    # and build a train loader
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # load the vae checkpoint
    vae, _ = load_vae_checkpoint(
        checkpoint_path = os.path.join(checkpoint_dir, trained_vae_filename),
        embedding_dim = embedding_dim,
        num_classes=num_classes,
        device=device
    )

    # instantiate model
    discrim = Discriminator(num_classes=num_classes)
    discrim_params = list(discrim.parameters())
    discrim_optimizer = torch.optim.Adam(discrim_params, lr=discrim_lr)

    vae_params = list(vae.decoder.parameters())
    vae_optimizer = torch.optim.Adam(vae_params, lr=decoder_lr)


    #finetune_VAE_GAN(vae, discrim, num_epochs, train_loader, vae_optimizer, discrim_optimizer, beta, lambda_adv, device, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()