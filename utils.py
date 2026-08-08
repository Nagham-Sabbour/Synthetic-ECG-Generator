import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset

from model import VAE

class BalancedBatchSampler(Sampler):
    '''Create batches with an equal number of samples from every class.'''

    def __init__(self, label_ids, batch_size, seed=42):
        self.class_ids = torch.unique(label_ids).tolist()
        self.num_classes = len(self.class_ids)

        assert batch_size % self.num_classes == 0, (
            f"batch_size must be divisible by {self.num_classes} classes"
        )

        self.samples_per_class = batch_size // self.num_classes
        self.num_batches = len(label_ids) // batch_size
        self.seed = seed
        self.epoch = 0

        self.class_indices = {
            class_id: torch.where(label_ids == class_id)[0].tolist()
            for class_id in self.class_ids
        }

    def __iter__(self):
        # Different order every epoch, but reproducible across complete runs.
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {}

        for class_id, indices in self.class_indices.items():
            pools[class_id] = indices.copy()
            rng.shuffle(pools[class_id])

        for _ in range(self.num_batches):
            batch = []

            for class_id in self.class_ids:
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


def load_preprocessing_metadata(data_root):
    '''Load class names and normalization values from preprocessing.'''
    mapping_path = os.path.join(data_root, "label_mapping.json")

    with open(mapping_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_balanced_train_loader(data_root, batch_size, seed=42):
    '''
    Load train.npz and return a strict class-balanced DataLoader.

    The model receives one-hot labels.
    The sampler uses integer label IDs.
    '''
    train_data = np.load(os.path.join(data_root, "train.npz"))

    signals = torch.from_numpy(train_data["signals"]).float().unsqueeze(1)
    label_ids = torch.from_numpy(train_data["labels"]).long()
    labels_onehot = torch.from_numpy(train_data["labels_onehot"]).float()

    assert signals.ndim == 3
    assert signals.shape[1] == 1
    assert len(signals) == len(label_ids) == len(labels_onehot)

    assert labels_onehot.ndim == 2
    assert torch.equal(labels_onehot.argmax(dim=1), label_ids)

    dataset = TensorDataset(signals, labels_onehot)

    batch_sampler = BalancedBatchSampler(
        label_ids=label_ids,
        batch_size=batch_size,
        seed=seed,
    )

    return DataLoader(dataset, batch_sampler=batch_sampler)


def load_vae_checkpoint(checkpoint_path, embedding_dim, num_classes, device='cpu'):
    '''
    Load a pretrained VAE from a checkpoint file saved by train_VAE.

    Args:
        checkpoint_path: path to the .pt checkpoint file
        embedding_dim: latent dimension used when the VAE was originally trained
        num_classes: number of classes used when the VAE was originally trained
        device: 'cpu' or 'cuda'
    
    Returns:
        vae: VAE instance with loaded weights, moved to device
        checkpoint: full checkpoint dict, in case epoch/loss info is needed
    '''

    vae = VAE(embedding_dim=embedding_dim, num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        loss = checkpoint.get("loss")
    elif "vae_state_dict" in checkpoint:
        state_dict = checkpoint["vae_state_dict"]
        loss = checkpoint.get("recon_loss")
    else:
        raise KeyError("Checkpoint does not contain VAE weights.")

    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()

    print(f"Loaded VAE checkpoint from {checkpoint_path} "
          f"(epoch={checkpoint.get('epoch')}, loss={loss})")

    return vae, checkpoint

def generate_conditioned_signals(vae, label_ids, embedding_dim, num_classes, device):
    """Generate normalized ECG segments for specified class IDs."""
    vae.eval()

    label_ids = torch.as_tensor(label_ids, dtype=torch.long, device=device)

    labels_onehot = F.one_hot(
        label_ids,
        num_classes=num_classes,
    ).float()

    z = torch.randn(
        len(label_ids),
        embedding_dim,
        device=device,
    )

    with torch.no_grad():
        generated = vae.decoder(z, labels_onehot)

    return generated.cpu()