import torch
import torch.nn.functional as F

import argparse
import os
import datetime

from model import Discriminator
from utils import load_vae_checkpoint, create_balanced_train_loader, load_preprocessing_metadata, create_val_loader
from visualize import generate_and_plot_samples, reconstruct_and_plot_samples, plot_training_losses

DATA_ROOT = "./processed_data"


def validate_VAE_GAN(vae, discriminator, val_loader, lambda_adv=1.0, device='cpu'):
    '''
    Run one pass over the validation set with no gradient updates.
    
    Args:
        vae: instance of model.VAE
        discriminator: instance of model.Discriminator
        val_loader: contains the validation set data
        lambda_adv: weight factor for adversarial loss term
        device: 'cpu' or 'cuda'

    Returns:
        avg_generator_loss, avg_recon_loss, avg_generator_adversarial_loss, avg_discrim_loss
    '''
    vae.eval()
    discriminator.eval()

    # initiate variables
    total_generator_loss = 0
    recon_loss_total = 0
    generator_adversarial_loss_total = 0
    discrim_loss_total = 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            # DISCRIMINATOR:
            
            with torch.no_grad():
                mean, logvar, recon_out = vae(signals, labels) 

            # Discriminator loss
            real_logits = discriminator(signals, labels)
            fake_logits = discriminator(recon_out.detach(), labels)
            real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            discrim_loss = real_loss + fake_loss

            # Update losses
            discrim_loss_total += discrim_loss.item()

            # GENERATOR (DECODER):
            
            mean, logvar, recon_out = vae(signals, labels) 

            # Compute reconstruction loss
            recon_loss = F.mse_loss(recon_out, signals, reduction='sum')

            # Compute generator adversarial loss
            fake_logits_for_gen = discriminator(recon_out, labels)
            generator_adversarial_loss = F.binary_cross_entropy_with_logits(fake_logits_for_gen, torch.ones_like(fake_logits_for_gen))

            # Overall generator loss (reconstruction loss + adversarial loss)
            # KL loss was not included because the encoder is frozen so it would remain constant
            generator_loss = recon_loss + (lambda_adv * generator_adversarial_loss)

            # Update losses
            total_generator_loss += generator_loss.item()
            recon_loss_total += recon_loss.item()
            generator_adversarial_loss_total += generator_adversarial_loss.item()

    num_batches = len(val_loader)
    avg_generator_loss = total_generator_loss / num_batches
    avg_recon_loss = recon_loss_total / num_batches
    avg_generator_adversarial_loss = generator_adversarial_loss_total / num_batches
    avg_discrim_loss = discrim_loss_total / num_batches

    return avg_generator_loss, avg_recon_loss, avg_generator_adversarial_loss, avg_discrim_loss


def finetune_VAE_GAN(vae, discriminator, num_epochs, train_loader, val_loader, vae_optimizer, discrim_optimizer, lambda_adv=1.0, device='cpu', checkpoint_dir='checkpoints', plots_dir='training_plots'):
    '''
    Fine-tune a pretrained VAE's decoder using a GAN-based discriminator.

    Args:
        vae: pretrained instance of model.VAE
        discriminator: instance of model.Discriminator
        num_epochs: number of epochs to fine-tune for
        train_loader: contains the training data
        val_loader: contains the validation data
        vae_optimizer: optimizer to use for training the VAE decoder
        discrim_optimizer: optimizer to use for training the GAN discriminator
        lambda_adv: weight factor for adversarial loss term
        device: 'cpu' or 'cuda'
        checkpoint_dir: directory name of where to save the model checkpoints
        plots_dir: directory name of where to save the training curve plots

    Returns:
        Best checkpoint path and final checkpoint path
    '''

    vae.to(device)
    discriminator.to(device)

    # freeze vae encoder during fine tuning
    for param in vae.encoder.parameters():
        param.requires_grad = False

    # for checkpoint saving
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    best_generator_loss = float('inf')
    best_checkpoint_path = None

    # for training plots: track per-epoch loss averages across whole run 
    total_generator_loss_history = []
    recon_loss_history = []
    generator_adversarial_loss_history = []
    discrim_loss_history = []
    val_generator_loss_history = []
    val_recon_loss_history = []
    val_generator_adversarial_loss_history = []
    val_discrim_loss_history = []
    
    for epoch in range(num_epochs):

        # put to train mode
        vae.train()
        vae.encoder.eval()
        discriminator.train()

        # initiate variables
        total_generator_loss = 0
        recon_loss_total = 0
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

            # Compute generator adversarial loss
            fake_logits_for_gen = discriminator(recon_out, labels)
            generator_adversarial_loss = F.binary_cross_entropy_with_logits(fake_logits_for_gen, torch.ones_like(fake_logits_for_gen))

            # Overall generator loss (reconstruction loss + adversarial loss)
            # KL loss was not included because the encoder is frozen so it would remain constant
            generator_loss = recon_loss + (lambda_adv * generator_adversarial_loss)

            # Backpropagate and optimize
            vae_optimizer.zero_grad()
            generator_loss.backward()
            vae_optimizer.step()

            # Update losses
            total_generator_loss += generator_loss.item()
            recon_loss_total += recon_loss.item()
            generator_adversarial_loss_total += generator_adversarial_loss.item()
                

        num_batches = len(train_loader)
        avg_generator_loss = total_generator_loss / num_batches
        avg_recon_loss = recon_loss_total / num_batches
        avg_generator_adversarial_loss = generator_adversarial_loss_total / num_batches
        avg_discrim_loss = discrim_loss_total / num_batches

        # run validation
        avg_val_generator_loss, avg_val_recon_loss, avg_val_generator_adversarial_loss, avg_val_discrim_loss = validate_VAE_GAN(vae, discriminator, val_loader, lambda_adv=lambda_adv, device=device)

        # append losses to history
        total_generator_loss_history.append(avg_generator_loss)
        recon_loss_history.append(avg_recon_loss)
        generator_adversarial_loss_history.append(avg_generator_adversarial_loss)
        discrim_loss_history.append(avg_discrim_loss)
        val_generator_loss_history.append(avg_val_generator_loss)
        val_recon_loss_history.append(avg_val_recon_loss)
        val_generator_adversarial_loss_history.append(avg_val_generator_adversarial_loss)
        val_discrim_loss_history.append(avg_val_discrim_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
            f"Generator Loss: {avg_generator_loss:.3f}  "
            f"Reconstruction Loss: {avg_recon_loss:.3f}  "
            f"Generator Adversarial Loss: {avg_generator_adversarial_loss:.3f}  "
            f"Discriminator Loss: {avg_discrim_loss:.3f}")

        # save best checkpoint by avg val loss
        if avg_val_generator_loss < best_generator_loss:
            best_generator_loss = avg_val_generator_loss
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

    print(f"Saved best checkpoint (generator_loss={best_generator_loss:.3f}) to {best_checkpoint_path}")
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    # plot loss curves
    plot_training_losses({
        'Total Generator Loss': total_generator_loss_history,
        'Reconstruction Loss': recon_loss_history,
        'Generator Adversarial Loss': generator_adversarial_loss_history,
        'Discriminator Loss': discrim_loss_history,
    }, output_dir=plots_dir, filename_prefix='vaegan_finetuning_loss')

    plot_training_losses({
        'Total Generator Loss (Val)': val_generator_loss_history,
        'Reconstruction Loss (Val)': val_recon_loss_history,
        'Generator Adversarial Loss (Val)': val_generator_adversarial_loss_history,
        'Discriminator Loss (Val)': val_discrim_loss_history,
    }, output_dir=plots_dir, filename_prefix='vaegan_validation_loss')

    return best_checkpoint_path, final_checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Synthetic ECG Generator (VAE + GAN)")
    parser.add_argument("--trained-vae-filename", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=66)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--decoder-lr", type=float, default=1e-3)
    parser.add_argument("--discrim-lr", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=32) #Note: must match the original vae training run
    parser.add_argument("--loss-lambda-adv", type=float, default=50)
    parser.add_argument("--num-classes", type=int, default=11) #Note: must match the original vae training run
    parser.add_argument("--checkpoint-dir", type=str, default='checkpoints')
    parser.add_argument("--visuals-dir", type=str, default='visuals')
    parser.add_argument("--plots-dir", type=str, default='training_plots')

    args = parser.parse_args()
    
    # set the arguments
    trained_vae_filename = args.trained_vae_filename
    embedding_dim = args.embedding_dim
    num_classes = args.num_classes
    decoder_lr = args.decoder_lr
    discrim_lr = args.discrim_lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    lambda_adv = args.loss_lambda_adv
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.checkpoint_dir
    visuals_dir = args.visuals_dir
    plots_dir = args.plots_dir

    # load the preprocessed dataset
    preprocessing_info = load_preprocessing_metadata(args.data_root)

    mean = preprocessing_info["signal_mean"]
    std = preprocessing_info["signal_std"]
    class_names = preprocessing_info["class_names"]
    num_classes = preprocessing_info["num_classes"]

    assert num_classes == args.num_classes, (
        f"Expected {args.num_classes} classes, but preprocessing produced "
        f"{num_classes} classes."
    )

    train_loader = create_balanced_train_loader(
        data_root=args.data_root,
        batch_size=batch_size,
        seed=42,
    )

    # load validation set
    val_loader = create_val_loader(data_root=args.data_root, batch_size=batch_size)

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


    best_checkpoint_path, _ = finetune_VAE_GAN(vae, discrim, num_epochs, train_loader, val_loader, vae_optimizer, discrim_optimizer, lambda_adv, device, checkpoint_dir=checkpoint_dir, plots_dir=plots_dir)

    # visualize results from the best trained model
    best_vae, _ = load_vae_checkpoint(
        checkpoint_path=best_checkpoint_path,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        device=device,
    )
    generate_and_plot_samples(best_vae, mean, std, num_classes=num_classes, samples_per_class=1, output_dir=visuals_dir, filename_prefix='vae_gan_generated', class_names=class_names, embedding_dim=embedding_dim, device=device)
    signals, labels = next(iter(val_loader))
    reconstruct_and_plot_samples(best_vae, signals, labels, mean, std, num_samples=6, output_dir=visuals_dir, filename_prefix='vae_gan_reconstructed', class_names=class_names, device=device)
    

if __name__ == "__main__":
    main()