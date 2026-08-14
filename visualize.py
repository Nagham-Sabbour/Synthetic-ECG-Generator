import datetime
import os

import matplotlib.pyplot as plt
import torch


def generate_and_plot_samples(vae, mean, std, num_classes=11, samples_per_class=1, output_dir='visuals', filename_prefix='generated_samples', class_names=None, embedding_dim=32, device='cpu'):
    '''
    Generate one or more normalized ECG samples for every class

    Args:
        vae: trained instance of model.VAE
        mean: scalar mean used for normalization during preprocessing
        std: scalar std used for normalization during preprocessing
        num_classes: number of diagnostic classes the model was trained with
        samples_per_class: number of samples to generate per class
        output_dir: folder name to save the plot to
        filename_prefix: prefix for the saved plot filename, timestamp will be appended
        class_names: optional list of class label strings (length num_classes) used for subplot titles
        embedding_dim: latent dimension the VAE was trained with
        device: 'cpu' or 'cuda'

    Returns: 
        save_path: path to the saved plot
    '''

    vae.to(device)
    vae.eval()

    fig, axes = plt.subplots(
        num_classes,
        samples_per_class,
        figsize=(4 * samples_per_class, 2 * num_classes),
        squeeze=False,
    )

    with torch.no_grad():
        for class_id in range(num_classes):
            labels = torch.zeros(samples_per_class, num_classes, device=device)
            labels[:, class_id] = 1.0
            latent = torch.randn(samples_per_class, embedding_dim, device=device)
            generated = vae.decoder(latent, labels)
            generated = (generated * std + mean).cpu().numpy()

            for sample_id in range(samples_per_class):
                axis = axes[class_id, sample_id]
                axis.plot(generated[sample_id, 0])

                if sample_id == 0:
                    class_name = class_names[class_id] if class_names else f"Class {class_id}"
                    axis.set_ylabel(class_name, fontsize=9)

                axis.set_xticks([])
                axis.set_yticks([])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved generated samples plot to {save_path}")
    return save_path



def reconstruct_and_plot_samples(vae, signals, labels, mean, std, num_samples=6, output_dir='visuals', filename_prefix='reconstructions', class_names=None, device='cpu'):
    '''
    Plot real ECGs beside their VAE reconstructions

    Args:
        vae: trained instance of model.VAE
        signals: batch of real signals, shape (batch, 1, 500)
        labels: corresponding one-hot labels, shape (batch, num_classes)
        mean: scalar mean used for normalization during preprocessing
        std: scalar std used for normalization during preprocessing
        num_samples: number of samples from the batch to plot (uses first num_samples in batch)
        output_dir: folder name to save the plot to
        filename_prefix: prefix for the saved plot filename, timestamp will be appended
        class_names: optional list of class label strings 
        device: 'cpu' or 'cuda'

    Returns: 
        save_path: path to the saved plot
    '''

    vae.to(device)
    vae.eval()

    signals = signals.to(device)
    labels = labels.to(device)
    num_samples = min(num_samples, signals.size(0))

    with torch.no_grad():
        _, _, reconstructions = vae(signals, labels)

    real_signals = (signals * std + mean).cpu().numpy()
    reconstructed_signals = (reconstructions * std + mean).cpu().numpy()
    label_ids = labels.argmax(dim=1).cpu()

    fig, axes = plt.subplots(
        num_samples,
        1,
        figsize=(8, 2 * num_samples),
        squeeze=False,
    )

    for sample_id in range(num_samples):
        axis = axes[sample_id, 0]
        axis.plot(real_signals[sample_id, 0], label="Original", linewidth=1.5)
        axis.plot(
            reconstructed_signals[sample_id, 0],
            label="Reconstructed",
            linewidth=1.0,
            linestyle="--",
        )

        class_id = label_ids[sample_id].item()
        class_name = class_names[class_id] if class_names else f"Class {class_id}"
        axis.set_ylabel(class_name, fontsize=9)

        if sample_id == 0:
            axis.legend(loc="upper right", fontsize=8)

        axis.set_xticks([])
        axis.set_yticks([])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reconstruction plot to {save_path}")
    return save_path


def plot_training_losses(losses_dict, output_dir='training_plots', filename_prefix='training_loss'):
    '''
    Plot training loss curves over epochs and save to folder.

    Args: 
        losses_dict: dict mapping loss name to list of per-epoch average values
                     eg. {'Total Loss': [...], 'Reconstruction Loss': [...], 'KL Divergence Loss': [...]}
        output_dir: folder name to save the plot to
        filename_prefix: prefix for the saved plot filename, timestamp will be appended

    Returns: 
        save_path: path to the saved plot
    '''

    num_losses = len(losses_dict)
    fig, axes = plt.subplots(
        num_losses,
        1,
        figsize=(8, 3 * num_losses),
        sharex=True,
        squeeze=False,
    )

    for index, (loss_name, values) in enumerate(losses_dict.items()):
        axis = axes[index, 0]
        axis.plot(range(1, len(values) + 1), values)
        axis.set_ylabel(loss_name, fontsize=9)
        axis.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Epoch")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved training loss plot to {save_path}")
    return save_path