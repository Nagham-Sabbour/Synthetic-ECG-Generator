import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import welch
import torch

from utils import (
    generate_conditioned_signals,
    load_preprocessing_metadata,
    load_vae_checkpoint,
)
from visualize import generate_and_plot_samples


def create_balanced_real_and_generated_sets(
    test_data,
    model,
    embedding_dim,
    num_classes,
    samples_per_class,
    device,
    seed,
):
    """Select real test samples and generate the same number for each class."""
    rng = np.random.default_rng(seed)

    real_signals = []
    real_labels = []
    generated_signals = []
    generated_labels = []

    test_labels = test_data["labels"]

    for class_id in range(num_classes):
        class_indices = np.where(test_labels == class_id)[0]

        if len(class_indices) == 0:
            print(f"Skipping class {class_id}: no test samples found.")
            continue

        num_samples = min(samples_per_class, len(class_indices))

        selected_indices = rng.choice(
            class_indices,
            size=num_samples,
            replace=False,
        )

        class_real_signals = test_data["signals"][selected_indices]

        class_generated_signals = generate_conditioned_signals(
            vae=model,
            label_ids=np.full(num_samples, class_id),
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            device=device,
        ).squeeze(1).numpy()

        real_signals.append(class_real_signals)
        real_labels.append(np.full(num_samples, class_id))

        generated_signals.append(class_generated_signals)
        generated_labels.append(np.full(num_samples, class_id))

    return (
        np.concatenate(real_signals),
        np.concatenate(real_labels),
        np.concatenate(generated_signals),
        np.concatenate(generated_labels),
    )


def compute_mean_psd(signals, sampling_rate):
    """Compute the mean PSD for a collection of ECG signals."""
    psd_values = []

    for signal in signals:
        frequencies, psd = welch(
            signal,
            fs=sampling_rate,
            nperseg=min(256, len(signal)),
        )
        psd_values.append(psd)

    return frequencies, np.mean(psd_values, axis=0)


def plot_psd_by_class(
    real_signals,
    real_labels,
    generated_signals,
    generated_labels,
    class_names,
    sampling_rate,
    output_path,
):
    """Plot real and generated PSDs separately for every diagnostic class."""
    num_classes = len(class_names)
    num_rows = int(np.ceil(num_classes / 2))

    fig, axes = plt.subplots(
        num_rows,
        2,
        figsize=(14, 3 * num_rows),
        squeeze=False,
    )

    psd_scores = []

    for class_id in range(num_classes):
        axis = axes[class_id // 2, class_id % 2]

        class_real = real_signals[real_labels == class_id]
        class_generated = generated_signals[generated_labels == class_id]

        frequencies, real_psd = compute_mean_psd(
            class_real,
            sampling_rate,
        )

        _, generated_psd = compute_mean_psd(
            class_generated,
            sampling_rate,
        )

        # Difference between real and generated frequency content.
        psd_log_mse = np.mean(
            (
                np.log10(real_psd + 1e-12)
                - np.log10(generated_psd + 1e-12)
            ) ** 2
        )

        psd_scores.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "log_psd_mse": psd_log_mse,
            }
        )

        axis.semilogy(
            frequencies,
            real_psd,
            color="black",
            label="Real",
        )

        axis.semilogy(
            frequencies,
            generated_psd,
            color="tab:blue",
            label="Generated",
        )

        axis.set_title(class_names[class_id])
        axis.set_xlim(0, 40)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD")
        axis.grid(True, alpha=0.3)
        axis.legend()

    # Hide any unused subplot.
    for index in range(num_classes, num_rows * 2):
        axes[index // 2, index % 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(psd_scores)


def detect_r_peak_features(signal, sampling_rate):
    """
    Detect R peaks and calculate rhythm features.

    This measures R-peak plausibility, not true R-peak accuracy. Generated
    signals do not have ground-truth peak annotations to compare against.
    """
    try:
        _, peak_info = nk.ecg_peaks(
            signal,
            sampling_rate=sampling_rate,
        )

        peaks = peak_info["ECG_R_Peaks"]
        num_peaks = len(peaks)

        if num_peaks >= 2:
            rr_seconds = np.diff(peaks) / sampling_rate
            mean_rr_ms = np.mean(rr_seconds) * 1000
            mean_hr_bpm = 60 / np.mean(rr_seconds)
        else:
            mean_rr_ms = np.nan
            mean_hr_bpm = np.nan

        plausible = (
            num_peaks >= 2
            and 30 <= mean_hr_bpm <= 220
        )

        return num_peaks, mean_rr_ms, mean_hr_bpm, plausible

    except Exception:
        return 0, np.nan, np.nan, False


def evaluate_r_peaks(
    signals,
    labels,
    source_name,
    class_names,
    sampling_rate,
):
    """Compute R-peak plausibility metrics for a set of signals."""
    rows = []

    for sample_index, (signal, class_id) in enumerate(zip(signals, labels)):
        num_peaks, mean_rr_ms, mean_hr_bpm, plausible = (
            detect_r_peak_features(signal, sampling_rate)
        )

        rows.append(
            {
                "source": source_name,
                "sample_index": sample_index,
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "num_r_peaks": num_peaks,
                "mean_rr_ms": mean_rr_ms,
                "mean_hr_bpm": mean_hr_bpm,
                "plausible_rhythm": plausible,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated ECG signals"
    )

    parser.add_argument("--data-root", default="./processed_data")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", default="./results/evaluation")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_info = load_preprocessing_metadata(args.data_root)

    class_names = preprocessing_info["class_names"]
    num_classes = preprocessing_info["num_classes"]
    signal_mean = preprocessing_info["signal_mean"]
    signal_std = preprocessing_info["signal_std"]
    sampling_rate = preprocessing_info["sampling_rate_hz"]

    test_data = np.load(
        Path(args.data_root) / "test.npz"
    )

    vae, _ = load_vae_checkpoint(
        checkpoint_path=args.checkpoint_path,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        device=device,
    )

    # 1. Visual inspection: one generated ECG for every class.
    generate_and_plot_samples(
        vae=vae,
        mean=signal_mean,
        std=signal_std,
        num_classes=num_classes,
        samples_per_class=1,
        output_dir=output_dir,
        filename_prefix="generated_samples",
        class_names=class_names,
        embedding_dim=args.embedding_dim,
        device=device,
    )

    # Create equal-size real and generated sets for fair PSD and R-peak comparison.
    (
        real_normalized,
        real_labels,
        generated_normalized,
        generated_labels,
    ) = create_balanced_real_and_generated_sets(
        test_data=test_data,
        model=vae,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        samples_per_class=args.samples_per_class,
        device=device,
        seed=args.seed,
    )

    # Convert back to mV for physiological signal analysis.
    real_signals = real_normalized * signal_std + signal_mean
    generated_signals = generated_normalized * signal_std + signal_mean

    # 2. PSD comparison.
    psd_scores = plot_psd_by_class(
        real_signals=real_signals,
        real_labels=real_labels,
        generated_signals=generated_signals,
        generated_labels=generated_labels,
        class_names=class_names,
        sampling_rate=sampling_rate,
        output_path=output_dir / "psd_comparison_by_class.png",
    )

    psd_scores.to_csv(
        output_dir / "psd_scores_by_class.csv",
        index=False,
    )

    # 3. R-peak plausibility comparison.
    real_r_peaks = evaluate_r_peaks(
        signals=real_signals,
        labels=real_labels,
        source_name="Real",
        class_names=class_names,
        sampling_rate=sampling_rate,
    )

    generated_r_peaks = evaluate_r_peaks(
        signals=generated_signals,
        labels=generated_labels,
        source_name="Generated",
        class_names=class_names,
        sampling_rate=sampling_rate,
    )

    r_peak_results = pd.concat(
        [real_r_peaks, generated_r_peaks],
        ignore_index=True,
    )

    r_peak_results.to_csv(
        output_dir / "r_peak_metrics_per_signal.csv",
        index=False,
    )

    r_peak_summary = (
        r_peak_results
        .groupby(["source", "class_name"])
        .agg(
            samples=("sample_index", "count"),
            mean_r_peaks=("num_r_peaks", "mean"),
            mean_rr_ms=("mean_rr_ms", "mean"),
            mean_hr_bpm=("mean_hr_bpm", "mean"),
            plausible_rhythm_rate=("plausible_rhythm", "mean"),
        )
        .reset_index()
    )

    r_peak_summary["plausible_rhythm_rate"] = (
        100 * r_peak_summary["plausible_rhythm_rate"]
    ).round(2)

    r_peak_summary.to_csv(
        output_dir / "r_peak_summary_by_class.csv",
        index=False,
    )

    print("\nPSD scores:")
    print(psd_scores)

    print("\nR-peak plausibility summary:")
    print(r_peak_summary)

    print(f"\nSaved evaluation results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()