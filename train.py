import torch
import torch.nn.functional as F

import argparse
import os
import datetime

from model import VAE
from utils import create_balanced_train_loader, load_preprocessing_metadata, create_val_loader, load_vae_checkpoint
from visualize import generate_and_plot_samples, reconstruct_and_plot_samples, plot_training_losses

DATA_ROOT = "./processed_data"

def validate_VAE(vae, val_loader, beta=1.0, device='cpu'):
    '''
    Run one pass over the validation set with no gradient updates.
    
    Args:
        vae: instance of model.VAE
        val_loader: contains the validation set data
        beta: weight factor for KL divergence loss term
        device: 'cpu' or 'cuda'

    Returns:
        avg_loss, avg_recon_loss, avg_kl_loss
    '''

    vae.eval()

    # initiate variables
    total_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            mean, logvar, recon_out = vae(signals, labels) 
            
            # Compute reconstruction loss
            recon_loss = F.mse_loss(recon_out, signals, reduction='sum')

            # Compute KL divergence loss
            kl_loss = -0.5 * torch.sum(logvar + 1 - mean.pow(2) - logvar.exp())

            # Add KL divergence loss to reconstruction loss
            loss = recon_loss + beta * kl_loss

            # Update losses
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = recon_loss_total / num_batches
    avg_kl_loss = kl_loss_total / num_batches

    return avg_loss, avg_recon_loss, avg_kl_loss


def train_VAE(vae, num_epochs, train_loader, val_loader, optimizer, beta=1.0, device='cpu', checkpoint_dir='checkpoints', plots_dir='training_plots'):
    '''
    Train the provided VAE model.

    Args:
        vae: instance of model.VAE
        num_epochs: number of epochs to train for
        train_loader: contains the training data
        val_loader: contains the validation data
        optimizer: optimizer to use for training
        beta: weight factor for KL divergence loss term
        device: 'cpu' or 'cuda'
        checkpoint_dir: directory name of where to save the model checkpoints
        plots_dir: directory name of where to save the training curve plots

    Returns:
        Best checkpoint path and final checkpoint path
    '''

    vae.to(device)

    # for checkpoint saving
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    best_loss = float('inf')
    best_checkpoint_path = None

    # for training plots: track per-epoch loss averages across whole run 
    total_loss_history = []
    recon_loss_history = []
    kl_loss_history = []
    val_loss_history = []
    val_recon_loss_history = []
    val_kl_loss_history = []

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

        # run validation
        avg_val_loss, avg_val_recon_loss, avg_val_kl_loss = validate_VAE(vae, val_loader, beta=beta, device=device)

        # append losses to history
        total_loss_history.append(avg_loss)
        recon_loss_history.append(avg_recon_loss)
        kl_loss_history.append(avg_kl_loss)
        val_loss_history.append(avg_val_loss)
        val_recon_loss_history.append(avg_val_recon_loss)
        val_kl_loss_history.append(avg_val_kl_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
            f"Total Loss: {avg_loss:.3f}  "
            f"Reconstruction Loss: {avg_recon_loss:.3f}  "
            f"KL Divergence Loss: {avg_kl_loss:.3f}")

        # save best checkpoint by avg val loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
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

    # plot loss curves
    plot_training_losses({
        'Total Loss': total_loss_history,
        'Reconstruction Loss': recon_loss_history,
        'KL Divergence Loss': kl_loss_history,
    }, output_dir=plots_dir, filename_prefix='vae_training_loss')

    plot_training_losses({
        'Total Loss (Val)': val_loss_history,
        'Reconstruction Loss (Val)': val_recon_loss_history,
        'KL Divergence Loss (Val)': val_kl_loss_history,
    }, output_dir=plots_dir, filename_prefix='vae_validation_loss')

    return best_checkpoint_path, final_checkpoint_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Synthetic ECG Generator (VAE)")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=66) # the batch size should be divisible by the num of classes 11
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--loss-beta", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--checkpoint-dir", type=str, default='checkpoints')
    parser.add_argument("--visuals-dir", type=str, default='visuals')
    parser.add_argument("--plots-dir", type=str, default='training_plots')

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
    plots_dir = args.plots_dir

    # load the training data
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

    # instantiate model
    vae = VAE(embedding_dim=embedding_dim, num_classes=num_classes)
    params = list(vae.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    best_checkpoint_path, _ = train_VAE(vae, num_epochs, train_loader, val_loader, optimizer, beta=beta, device=device, checkpoint_dir=checkpoint_dir, plots_dir=plots_dir)

    # visualize results from the best trained model
    best_vae, _ = load_vae_checkpoint(
        checkpoint_path=best_checkpoint_path,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        device=device,
    )
    generate_and_plot_samples(best_vae, mean, std, num_classes=num_classes, samples_per_class=1, output_dir=visuals_dir, filename_prefix='vae_generated', class_names=class_names, embedding_dim=embedding_dim, device=device)
    signals, labels = next(iter(val_loader))
    reconstruct_and_plot_samples(best_vae, signals, labels, mean, std, num_samples=6, output_dir=visuals_dir, filename_prefix='vae_reconstructed', class_names=class_names, device=device)


if __name__ == "__main__":
    main()
