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

from model import VAE
from visualize import generate_and_plot_samples, reconstruct_and_plot_samples

DATA_ROOT = os.path.expanduser(
    "./data/ptb-xl-preprocessed-train"
)

def train_VAE(vae, num_epochs, train_loader, optimizer, beta=1.0, device='cpu', checkpoint_dir='checkpoints'):
    '''
    Train the provided VAE model.

    Args:
        vae: instance of model.VAE
        num_epochs: number of epochs to train for
        train_loader: contains the training data
        optimizer: optimizer to use for training
        beta: weight factor for KL divergence loss term
        device: 'cpu' or 'cuda'
        checkpoint_dir: directory name of where to save the model checkpoints
    '''

    vae.to(device)

    # for checkpoint saving
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    best_loss = float('inf')
    best_checkpoint_path = None

    for epoch in range(num_epochs):

        # put to train mode
        vae.train()

        # initiate variables
        total_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0

        # train loop
        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            mean, logvar, recon_out = vae(signals, labels) 

            # Compute reconstruction loss
            recon_loss = F.mse_loss(recon_out, signals, reduction='sum')

            # Compute KL divergence loss
            kl_loss = -0.5 * torch.sum(logvar + 1 - mean.pow(2) - logvar.exp())

            # Add KL divergence loss to reconstruction loss
            loss = recon_loss + beta * kl_loss

            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update losses
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = recon_loss_total / num_batches
        avg_kl_loss = kl_loss_total / num_batches

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
            f"Total Loss: {avg_loss:.3f}  "
            f"Reconstruction Loss: {avg_recon_loss:.3f}  "
            f"KL Divergence Loss: {avg_kl_loss:.3f}")

        # save best checkpoint by avg train loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, f"vae_best_{run_timestamp}.pt")
            torch.save({
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }, best_checkpoint_path)

    # save final checkpoint regardless of loss
    final_checkpoint_path = os.path.join(checkpoint_dir, f"vae_final_epoch{num_epochs}_{run_timestamp}.pt")
    torch.save({
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'loss': avg_loss,
    }, final_checkpoint_path)

    print(f"Saved best checkpoint (loss={best_loss:.3f}) to {best_checkpoint_path}")
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    return best_checkpoint_path, final_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Synthetic ECG Generator (VAE)")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--loss-beta", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--checkpoint-dir", type=str, default='checkpoints')
    parser.add_argument("--visuals-dir", type=str, default='visuals')

    args = parser.parse_args()

    # set the arguments
    embedding_dim = args.embedding_dim
    num_classes = args.num_classes
    lr = args.lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    beta = args.loss_beta
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.checkpoint_dir
    visuals_dir = args.visuals_dir

    # load the preprocessed dataset - TODO
    #train_dataset = 
    # and build a train loader
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # instantiate model
    vae = VAE(embedding_dim=embedding_dim, num_classes=num_classes)
    params = list(vae.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    #train_VAE(vae, num_epochs, train_loader, optimizer, beta=beta, device=device, checkpoint_dir=checkpoint_dir)

    # TODO - load mean/std and class names from preprocessing
    #mean = 
    #std = 
    #class_names = 

    # visualize results from the trained model
    #generate_and_plot_samples(vae, mean, std, num_classes=num_classes, samples_per_class=1, output_dir=visuals_dir, filename_prefix='vae_generated', class_names=class_names, embedding_dim=embedding_dim, device=device)
    #signals, labels = next(iter(train_loader))
    #reconstruct_and_plot_samples(vae, signals, labels, mean, std, num_samples=6, output_dir=visuals_dir, filename_prefix='vae_reconstructed', class_names=class_names, device=device)


if __name__ == "__main__":
    main()


