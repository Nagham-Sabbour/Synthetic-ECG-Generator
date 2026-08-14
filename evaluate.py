import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import wasserstein_distance
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
    '''Select real test samples and generate the same number for each class.'''
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


def plot_real_vs_generated_by_class(
    real_signals,
    real_labels,
    generated_signals,
    generated_labels,
    class_names,
    sampling_rate,
    output_path,
):
    '''Plot one real and one generated ECG for each class.'''
    fig, axes = plt.subplots(
        len(class_names),
        2,
        figsize=(12, 2.2 * len(class_names)),
        squeeze=False,
    )

    for class_id, class_name in enumerate(class_names):
        real_signal = real_signals[real_labels == class_id][0]
        generated_signal = generated_signals[
            generated_labels == class_id
        ][0]

        time = np.arange(len(real_signal)) / sampling_rate

        axes[class_id, 0].plot(time, real_signal, color="black")
        axes[class_id, 0].set_title(f"{class_name}: Real")
        axes[class_id, 0].set_ylabel("mV")
        axes[class_id, 0].grid(alpha=0.25)

        axes[class_id, 1].plot(
            time,
            generated_signal,
            color="tab:blue",
        )
        axes[class_id, 1].set_title(f"{class_name}: Generated")
        axes[class_id, 1].grid(alpha=0.25)

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_mean_psd(signals, sampling_rate):
    '''Compute the mean PSD for a collection of ECG signals.'''
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
    '''Plot real and generated PSDs separately for every diagnostic class.'''
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

        # Compare real and generated frequency content on a log scale
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

    # Hide an unused subplot when the number of classes is odd
    for index in range(num_classes, num_rows * 2):
        axes[index // 2, index % 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return pd.DataFrame(psd_scores)


def detect_r_peak_features(signal, sampling_rate):
    '''Detect R peaks and calculate simple rhythm features.'''
    try:
        _, peak_info = nk.ecg_peaks(
            signal,
            sampling_rate=sampling_rate,
        )

        peaks = peak_info["ECG_R_Peaks"]
        num_peaks = len(peaks)

        if num_peaks < 2:
            return num_peaks, np.nan, np.nan

        rr_seconds = np.diff(peaks) / sampling_rate
        mean_rr_ms = np.mean(rr_seconds) * 1000
        mean_hr_bpm = 60 / np.mean(rr_seconds)

        return num_peaks, mean_rr_ms, mean_hr_bpm

    except Exception:
        return 0, np.nan, np.nan


def build_rhythm_reference(
    validation_signals,
    validation_labels,
    num_classes,
    sampling_rate,
):
    '''Build class-specific plausible rhythm ranges from real validation ECGs.'''
    reference = {}

    for class_id in range(num_classes):
        class_signals = validation_signals[
            validation_labels == class_id
        ]

        peak_counts = []
        heart_rates = []

        for signal in class_signals:
            num_peaks, _, heart_rate = detect_r_peak_features(
                signal,
                sampling_rate,
            )

            if num_peaks >= 2 and np.isfinite(heart_rate):
                peak_counts.append(num_peaks)
                heart_rates.append(heart_rate)

        if not heart_rates:
            raise RuntimeError(
                f"No valid validation rhythm measurements for class {class_id}."
            )

        reference[class_id] = {
            "min_r_peaks": max(
                2,
                int(np.floor(np.percentile(peak_counts, 2.5))) - 1,
            ),
            "max_r_peaks": int(
                np.ceil(np.percentile(peak_counts, 97.5))
            ) + 1,
            "min_hr_bpm": max(
                40.0,
                float(np.percentile(heart_rates, 2.5)),
            ),
            "max_hr_bpm": min(
                180.0,
                float(np.percentile(heart_rates, 97.5)),
            ),
        }

    return reference


def rhythm_matches_reference(
    num_peaks,
    mean_hr_bpm,
    class_id,
    rhythm_reference,
):
    '''Check whether a rhythm fits the real validation distribution.'''
    if not np.isfinite(mean_hr_bpm):
        return False

    class_reference = rhythm_reference[int(class_id)]

    return (
        class_reference["min_r_peaks"] <= num_peaks
        <= class_reference["max_r_peaks"]
        and class_reference["min_hr_bpm"] <= mean_hr_bpm
        <= class_reference["max_hr_bpm"]
    )


def evaluate_r_peaks(
    signals,
    labels,
    source_name,
    class_names,
    sampling_rate,
    rhythm_reference,
):
    '''Evaluate R-peak features against class-specific real rhythm ranges.'''
    rows = []

    for sample_index, (signal, class_id) in enumerate(zip(signals, labels)):
        num_peaks, mean_rr_ms, mean_hr_bpm = detect_r_peak_features(
            signal,
            sampling_rate,
        )

        rhythm_match = rhythm_matches_reference(
            num_peaks=num_peaks,
            mean_hr_bpm=mean_hr_bpm,
            class_id=class_id,
            rhythm_reference=rhythm_reference,
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
                "rhythm_match": rhythm_match,
            }
        )

    return pd.DataFrame(rows)


def compare_rhythm_distributions(
    real_r_peaks,
    generated_r_peaks,
    class_names,
):
    '''Compare real and generated HR and R-peak-count distributions.'''
    rows = []

    for class_id, class_name in enumerate(class_names):
        real_class = real_r_peaks[
            real_r_peaks["class_id"] == class_id
        ]

        generated_class = generated_r_peaks[
            generated_r_peaks["class_id"] == class_id
        ]

        real_hr = real_class["mean_hr_bpm"].dropna().to_numpy()
        generated_hr = generated_class["mean_hr_bpm"].dropna().to_numpy()

        real_peaks = real_class["num_r_peaks"].to_numpy()
        generated_peaks = generated_class["num_r_peaks"].to_numpy()

        hr_distance = np.nan
        if len(real_hr) > 0 and len(generated_hr) > 0:
            hr_distance = wasserstein_distance(
                real_hr,
                generated_hr,
            )

        peak_count_distance = np.nan
        if len(real_peaks) > 0 and len(generated_peaks) > 0:
            peak_count_distance = wasserstein_distance(
                real_peaks,
                generated_peaks,
            )

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "hr_wasserstein_distance_bpm": hr_distance,
                "r_peak_count_wasserstein_distance": peak_count_distance,
            }
        )

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated ECG signals"
    )

    parser.add_argument("--data-root", default="./processed_data")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_name = Path(args.checkpoint_path).stem
    output_dir = Path(args.results_root) / checkpoint_name
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

    validation_data = np.load(
        Path(args.data_root) / "validation.npz"
    )

    vae, _ = load_vae_checkpoint(
        checkpoint_path=args.checkpoint_path,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        device=device,
    )

    # Save one generated ECG for each class
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

    # Use equal class sizes for fair PSD and R-peak comparisons
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

    # Convert normalized signals back to mV for physiological metrics
    real_signals = real_normalized * signal_std + signal_mean
    generated_signals = generated_normalized * signal_std + signal_mean
    validation_signals = validation_data["signals"] * signal_std + signal_mean

    plot_real_vs_generated_by_class(
        real_signals=real_signals,
        real_labels=real_labels,
        generated_signals=generated_signals,
        generated_labels=generated_labels,
        class_names=class_names,
        sampling_rate=sampling_rate,
        output_path=output_dir / "real_vs_generated_by_class.png",
    )

    rhythm_reference = build_rhythm_reference(
        validation_signals=validation_signals,
        validation_labels=validation_data["labels"],
        num_classes=num_classes,
        sampling_rate=sampling_rate,
    )

    rhythm_reference_table = pd.DataFrame(
        [
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                **values,
            }
            for class_id, values in rhythm_reference.items()
        ]
    )

    rhythm_reference_table.to_csv(
        output_dir / "rhythm_reference_ranges.csv",
        index=False,
    )

    # Save PSD comparisons for each class
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

    # Compare R-peak and heart-rate features with validation data
    real_r_peaks = evaluate_r_peaks(
        signals=real_signals,
        labels=real_labels,
        source_name="Real",
        class_names=class_names,
        sampling_rate=sampling_rate,
        rhythm_reference=rhythm_reference,
    )

    generated_r_peaks = evaluate_r_peaks(
        signals=generated_signals,
        labels=generated_labels,
        source_name="Generated",
        class_names=class_names,
        sampling_rate=sampling_rate,
        rhythm_reference=rhythm_reference,
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
            rhythm_match_rate=("rhythm_match", "mean"),
        )
        .reset_index()
    )

    r_peak_summary["rhythm_match_rate"] = (
        100 * r_peak_summary["rhythm_match_rate"]
    ).round(2)

    r_peak_summary.to_csv(
        output_dir / "r_peak_summary_by_class.csv",
        index=False,
    )

    rhythm_distances = compare_rhythm_distributions(
        real_r_peaks=real_r_peaks,
        generated_r_peaks=generated_r_peaks,
        class_names=class_names,
    )

    rhythm_distances.to_csv(
        output_dir / "rhythm_distribution_distances.csv",
        index=False,
    )

    print("\nPSD scores:")
    print(psd_scores)

    print("\nR-peak plausibility summary:")
    print(r_peak_summary)

    print(f"\nSaved evaluation results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
