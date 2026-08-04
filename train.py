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

from model import VAE

DATA_ROOT = os.path.expanduser(
    "./processed_data"
)

def train_VAE(vae, num_epochs, train_loader, optimizer, beta=1.0, device='cpu'):
    '''
    Train the provided VAE model.

    Args:
        vae: instance of model.VAE
        num_epochs: number of epochs to train for
        train_loader: contains the training data
        optimizer: optimizer to use for training
        beta: weight factor for KL divergence loss term
    '''

    vae.to(device)

    for epoch in range(num_epochs):

        # put to train mode
        vae.train()

        # initiate variables
        total_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0

        # train loop
        for signals, _ in train_loader:
            signals = signals.to(device)

            mean, logvar, recon_out = vae(signals) 

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
    parser = argparse.ArgumentParser(description="Train Synthetic ECG Generator")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=66) # the batch size should be divisible by the num of classes 11
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--loss-beta", type=float, default=1.0)

    args = parser.parse_args()

    # set the arguments
    embedding_dim = args.embedding_dim
    lr = args.lr
    num_epochs = args.epochs
    batch_size = math.ceil(args.batch_size / 11) * 11
    beta = args.loss_beta
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    vae = VAE(embedding_dim=embedding_dim)
    params = list(vae.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    train_VAE(vae, num_epochs, train_loader, optimizer, beta=beta, device=device)



if __name__ == "__main__":
    main()


