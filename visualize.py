import os
import torch
import matplotlib.pyplot as plt
import datetime

def generate_and_plot_samples(vae, mean, std, num_classes=11, samples_per_class=1, output_dir='visuals', filename_prefix='generated_samples', class_names=None, embedding_dim=32, device='cpu'):
    '''
    Generate synthetic ECG samples for each class and save the plots.

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

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(4*samples_per_class, 2*num_classes), squeeze=False)

    with torch.no_grad():
        for class_idx in range(num_classes):
            label = torch.zeros(samples_per_class, num_classes, device=device)
            label[:, class_idx] = 1.0

            z = torch.randn(samples_per_class, embedding_dim, device=device)
            generated = vae.decoder(z, label)

            generated = generated * std + mean
            generated = generated.cpu().numpy()

            for sample_idx in range(samples_per_class):
                ax = axes[class_idx, sample_idx]
                ax.plot(generated[sample_idx, 0, :])

                if sample_idx == 0:
                    label_text = class_names[class_idx] if class_names else f"Class {class_idx}"
                    ax.set_ylabel(label_text, fontsize=9)

                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Saved generated samples plot to {save_path}")

    return save_path



def reconstruct_and_plot_samples(vae, signals, labels, mean, std, num_samples=6, output_dir='visuals', filename_prefix='reconstructions', class_names=None, device='cpu'):
    '''
    Reconstruct real ECG samples through the VAE and save comparison plots (original vs. reconstructed).

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
        _, _, recon = vae(signals, labels)

    real = (signals * std + mean).cpu().numpy()
    recon = (recon * std + mean).cpu().numpy()
    labels_cpu = labels.cpu()

    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 2*num_samples), squeeze=False)

    for i in range(num_samples):
        ax = axes[i, 0]
        ax.plot(real[i, 0, :], label='Original', linewidth=1.5)
        ax.plot(recon[i, 0, :], label='Reconstructed', linewidth=1.0, linestyle='--')

        class_idx = torch.argmax(labels_cpu[i]).item()
        label_text = class_names[class_idx] if class_names else f"Class {class_idx}"
        ax.set_ylabel(label_text, fontsize=9)

        if i==0:
            ax.legend(loc='upper right', fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Saved reconstruction plot to {save_path}")

    return save_path