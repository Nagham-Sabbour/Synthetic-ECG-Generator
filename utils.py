import torch

from model import VAE

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
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.to(device)

    print(f"Loaded VAE checkpoint from {checkpoint_path} "
          f"(epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss'):.3f})")

    return vae, checkpoint