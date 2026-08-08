import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    generate_conditioned_signals,
    load_preprocessing_metadata,
    load_vae_checkpoint,
)


class ECGClassifier(nn.Module):
    """Small CNN classifier used only for the TSTR experiment."""

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, signals):
        features = self.features(signals)
        features = features.squeeze(-1)
        return self.classifier(features)


def select_balanced_real_training_set(
    train_data,
    num_classes,
    samples_per_class,
    seed,
):
    """Choose the same number of real training segments from every class."""
    rng = np.random.default_rng(seed)

    signals = []
    labels = []

    for class_id in range(num_classes):
        class_indices = np.where(train_data["labels"] == class_id)[0]

        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} real training "
                f"samples, fewer than requested {samples_per_class}."
            )

        selected = rng.choice(
            class_indices,
            size=samples_per_class,
            replace=False,
        )

        signals.append(train_data["signals"][selected])
        labels.append(np.full(samples_per_class, class_id))

    return np.concatenate(signals), np.concatenate(labels)


def generate_balanced_synthetic_training_set(
    vae,
    embedding_dim,
    num_classes,
    samples_per_class,
    device,
):
    """Generate the same number of synthetic ECGs for every class."""
    generated_signals = []
    generated_labels = []

    for class_id in range(num_classes):
        class_labels = np.full(
            samples_per_class,
            class_id,
        )

        generated = generate_conditioned_signals(
            vae=vae,
            label_ids=class_labels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            device=device,
        ).squeeze(1).numpy()

        generated_signals.append(generated)
        generated_labels.append(class_labels)

    return (
        np.concatenate(generated_signals),
        np.concatenate(generated_labels),
    )


def train_classifier(
    train_signals,
    train_labels,
    num_classes,
    epochs,
    batch_size,
    learning_rate,
    device,
):
    """Train a classifier on real or synthetic ECG signals."""
    dataset = TensorDataset(
        torch.from_numpy(train_signals).float().unsqueeze(1),
        torch.from_numpy(train_labels).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    classifier = ECGClassifier(num_classes).to(device)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0

        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            logits = classifier(signals)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Classifier epoch {epoch + 1}/{epochs} "
            f"- loss: {average_loss:.4f}"
        )

    return classifier


def evaluate_classifier(
    classifier,
    test_signals,
    test_labels,
    class_names,
    batch_size,
    device,
):
    """Evaluate a classifier on real test ECGs."""
    dataset = TensorDataset(
        torch.from_numpy(test_signals).float().unsqueeze(1),
        torch.from_numpy(test_labels).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    classifier.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for signals, labels in loader:
            logits = classifier(signals.to(device))
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(predicted_labels)
            targets.extend(labels.numpy())

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train-on-Synthetic-Test-on-Real experiment"
    )

    parser.add_argument("--data-root", default="./processed_data")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", default="./results/tstr")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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

    train_data = np.load(
        Path(args.data_root) / "train.npz"
    )

    test_data = np.load(
        Path(args.data_root) / "test.npz"
    )

    vae, _ = load_vae_checkpoint(
        checkpoint_path=args.checkpoint_path,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        device=device,
    )

    # Use the same balanced class count in both training experiments.
    real_train_signals, real_train_labels = (
        select_balanced_real_training_set(
            train_data=train_data,
            num_classes=num_classes,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    )

    synthetic_train_signals, synthetic_train_labels = (
        generate_balanced_synthetic_training_set(
            vae=vae,
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            samples_per_class=args.samples_per_class,
            device=device,
        )
    )

    # Baseline: train on real, test on real.
    print("\nTraining classifier on real ECGs...")
    real_classifier = train_classifier(
        train_signals=real_train_signals,
        train_labels=real_train_labels,
        num_classes=num_classes,
        epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    trtr_results = evaluate_classifier(
        classifier=real_classifier,
        test_signals=test_data["signals"],
        test_labels=test_data["labels"],
        class_names=class_names,
        batch_size=args.batch_size,
        device=device,
    )

    # Main TSTR experiment: train on synthetic, test on real.
    print("\nTraining classifier on synthetic ECGs...")
    synthetic_classifier = train_classifier(
        train_signals=synthetic_train_signals,
        train_labels=synthetic_train_labels,
        num_classes=num_classes,
        epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )

    tstr_results = evaluate_classifier(
        classifier=synthetic_classifier,
        test_signals=test_data["signals"],
        test_labels=test_data["labels"],
        class_names=class_names,
        batch_size=args.batch_size,
        device=device,
    )

    results = {
        "samples_per_class": args.samples_per_class,
        "classifier_epochs": args.classifier_epochs,
        "train_on_real_test_on_real": trtr_results,
        "train_on_synthetic_test_on_real": tstr_results,
    }

    with open(
        output_dir / "tstr_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\nTSTR results")
    print(f"TRTR accuracy: {trtr_results['accuracy']:.4f}")
    print(f"TRTR macro F1: {trtr_results['macro_f1']:.4f}")
    print(f"TSTR accuracy: {tstr_results['accuracy']:.4f}")
    print(f"TSTR macro F1: {tstr_results['macro_f1']:.4f}")

    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()