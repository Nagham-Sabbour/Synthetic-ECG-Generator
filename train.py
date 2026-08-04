import torch
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split, Sampler
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
    "./processed_data"
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

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, seed=42):
        self.labels = labels
        self.class_ids = torch.unique(labels).tolist()
        self.num_classes = len(self.class_ids)

        assert batch_size % self.num_classes == 0, (
            f"batch_size must be divisible by {self.num_classes} classes"
        )

        self.samples_per_class = batch_size // self.num_classes
        self.num_batches = len(labels) // batch_size
        self.seed = seed

        self.class_indices = {
            class_id: torch.where(labels == class_id)[0].tolist()
            for class_id in self.class_ids
        }

    def __iter__(self):
        rng = random.Random(self.seed)

        # Make a shuffled pool of indices for every class
        pools = {}
        for class_id, indices in self.class_indices.items():
            pools[class_id] = indices.copy()
            rng.shuffle(pools[class_id])

        for _ in range(self.num_batches):
            batch = []

            for class_id in self.class_ids:
                # Refill and reshuffle a class pool when it runs out
                while len(pools[class_id]) < self.samples_per_class:
                    extra_indices = self.class_indices[class_id].copy()
                    rng.shuffle(extra_indices)
                    pools[class_id].extend(extra_indices)

                batch.extend(pools[class_id][:self.samples_per_class])
                pools[class_id] = pools[class_id][self.samples_per_class:]

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Synthetic ECG Generator (VAE)")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=66) # the batch size should be divisible by the num of classes 11
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
    batch_size = math.ceil(args.batch_size / 11) * 11
    beta = args.loss_beta
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = args.checkpoint_dir
    visuals_dir = args.visuals_dir

    # Load the training data
    train_data = np.load(os.path.join(args.data_root, "train.npz"))

    signals = torch.from_numpy(train_data["signals"]).float()
    labels = torch.from_numpy(train_data["labels"]).long()

    # Change the shape for Conv1d input
    signals = signals.unsqueeze(1)
    
    # Confirm the right shape
    assert signals.ndim == 3
    assert signals.shape[1] == 1
    assert len(signals) == len(labels)

    train_dataset = TensorDataset(signals, labels)

    balanced_batch_sampler = BalancedBatchSampler(
        labels=labels,
        batch_size=batch_size,
        seed=42,
    )

    train_loader = DataLoader(train_dataset, batch_sampler=balanced_batch_sampler )

    # instantiate model
    vae = VAE(embedding_dim=embedding_dim, num_classes=num_classes)
    params = list(vae.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    train_VAE(vae, num_epochs, train_loader, optimizer, beta=beta, device=device, checkpoint_dir=checkpoint_dir)

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


