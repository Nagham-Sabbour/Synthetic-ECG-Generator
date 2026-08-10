import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    generate_conditioned_signals,
    load_preprocessing_metadata,
    load_vae_checkpoint,
)


class ECGClassifier(nn.Module):
    '''CNN classifier used only for the TSTR experiment.'''

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, signals):
        features = self.features(signals).squeeze(-1)
        return self.classifier(features)

def select_balanced_real_training_set(
    train_data,
    num_classes,
    samples_per_class,
    seed,
):
    '''Choose the same number of real training segments from every class.'''
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
    '''Generate the same number of synthetic ECGs for every class.'''
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
    validation_signals,
    validation_labels,
    num_classes,
    class_names,
    epochs,
    batch_size,
    learning_rate,
    patience,
    seed,
    device,
):
    '''Train a classifier and keep the best validation macro-F1 checkpoint.'''
    torch.manual_seed(seed)

    dataset = TensorDataset(
        torch.from_numpy(train_signals).float().unsqueeze(1),
        torch.from_numpy(train_labels).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    classifier = ECGClassifier(num_classes).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_state = None
    best_validation_f1 = -np.inf
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "validation_macro_f1": [],
    }

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            loss = F.cross_entropy(classifier(signals), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        validation_results = evaluate_classifier(
            classifier=classifier,
            test_signals=validation_signals,
            test_labels=validation_labels,
            class_names=class_names,
            batch_size=batch_size,
            device=device,
        )

        validation_f1 = validation_results["macro_f1"]

        history["train_loss"].append(average_loss)
        history["validation_macro_f1"].append(validation_f1)

        print(
            f"Classifier epoch {epoch + 1}/{epochs} | "
            f"train loss: {average_loss:.4f} | "
            f"validation macro F1: {validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_state = copy.deepcopy(classifier.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation macro F1: {best_validation_f1:.4f}"
            )
            break

    classifier.load_state_dict(best_state)
    return classifier, history


def evaluate_classifier(
    classifier,
    test_signals,
    test_labels,
    class_names,
    batch_size,
    device,
):
    '''Evaluate a classifier on real test ECGs.'''
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
        "targets": np.asarray(targets),
        "predictions": np.asarray(predictions),
    }

def plot_classifier_history(history, title, output_path):
    '''Save learning curves for the TSTR classifier.'''
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"])
    axes[0].set_title(f"{title}: Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["validation_macro_f1"])
    axes[1].set_title(f"{title}: Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_classifier_diagnostics(results, class_names, title, output_dir, prefix):
    '''Save a normalized confusion matrix and per-class metrics.'''
    targets = results["targets"]
    predictions = results["predictions"]

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        normalize="true",
    )

    fig, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format=".2f",
        colorbar=False,
    )

    axis.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{prefix}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    report = results["classification_report"]

    per_class = pd.DataFrame(
        [
            {
                "class_name": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1_score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            }
            for class_name in class_names
        ]
    )

    per_class.to_csv(
        output_dir / f"{prefix}_per_class_metrics.csv",
        index=False,
    )

def main():
    parser = argparse.ArgumentParser(
        description="Train-on-Synthetic-Test-on-Real experiment"
    )

    parser.add_argument("--data-root", default="./processed_data")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-epochs", type=int, default=80)
    parser.add_argument("--classifier-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

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

    train_data = np.load(
        Path(args.data_root) / "train.npz"
    )

    validation_data = np.load(
        Path(args.data_root) / "validation.npz"
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
    real_classifier, real_history = train_classifier(
        train_signals=real_train_signals,
        train_labels=real_train_labels,
        validation_signals=validation_data["signals"],
        validation_labels=validation_data["labels"],
        class_names=class_names,
        num_classes=num_classes,
        epochs=args.classifier_epochs,
        patience=args.classifier_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
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
    synthetic_classifier, synthetic_history = train_classifier(
        train_signals=synthetic_train_signals,
        train_labels=synthetic_train_labels,
        validation_signals=validation_data["signals"],
        validation_labels=validation_data["labels"],
        class_names=class_names,
        num_classes=num_classes,
        epochs=args.classifier_epochs,
        patience=args.classifier_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
    )

    tstr_results = evaluate_classifier(
        classifier=synthetic_classifier,
        test_signals=test_data["signals"],
        test_labels=test_data["labels"],
        class_names=class_names,
        batch_size=args.batch_size,
        device=device,
    )

    plot_classifier_history(
        real_history,
        title="TRTR Classifier",
        output_path=output_dir / "trtr_training_history.png",
    )

    plot_classifier_history(
        synthetic_history,
        title="TSTR Classifier",
        output_path=output_dir / "tstr_training_history.png",
    )

    save_classifier_diagnostics(
        results=trtr_results,
        class_names=class_names,
        title="TRTR: Train on Real, Test on Real",
        output_dir=output_dir,
        prefix="trtr",
    )

    save_classifier_diagnostics(
        results=tstr_results,
        class_names=class_names,
        title="TSTR: Train on Synthetic, Test on Real",
        output_dir=output_dir,
        prefix="tstr",
    )

    results = {
        "samples_per_class": args.samples_per_class,
        "classifier_epochs_requested": args.classifier_epochs,
        "classifier_patience": args.classifier_patience,
        "train_on_real_test_on_real": {
            "accuracy": trtr_results["accuracy"],
            "macro_f1": trtr_results["macro_f1"],
            "classification_report": trtr_results["classification_report"],
        },
        "train_on_synthetic_test_on_real": {
            "accuracy": tstr_results["accuracy"],
            "macro_f1": tstr_results["macro_f1"],
            "classification_report": tstr_results["classification_report"],
        },
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