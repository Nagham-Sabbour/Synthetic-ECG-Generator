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
    '''Small CNN used only for the TRTR and TSTR experiments'''

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
            
            # Summarize the time dimension before the final classifier
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, signals):
        features = self.features(signals).squeeze(-1)
        return self.classifier(features)


def select_balanced_real_training_set(train_data, num_classes, samples_per_class, seed):
    '''Randomly select the same number of real ECGs from each class'''
    rng = np.random.default_rng(seed)
    signals, labels = [], []

    for class_id in range(num_classes):
        class_indices = np.where(train_data["labels"] == class_id)[0]
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} real training "
                f"samples, fewer than requested {samples_per_class}."
            )
        selected = rng.choice(class_indices, size=samples_per_class, replace=False)
        signals.append(train_data["signals"][selected])
        labels.append(np.full(samples_per_class, class_id))

    return np.concatenate(signals), np.concatenate(labels)


def generate_balanced_synthetic_training_set(vae, embedding_dim, num_classes, samples_per_class, device):
    '''Generate the same number of synthetic ECGs for every class'''
    generated_signals, generated_labels = [], []

    for class_id in range(num_classes):
        class_labels = np.full(samples_per_class, class_id)
        generated = generate_conditioned_signals(
            vae=vae,
            label_ids=class_labels,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            device=device,
        ).squeeze(1).numpy()
        generated_signals.append(generated)
        generated_labels.append(class_labels)

    return np.concatenate(generated_signals), np.concatenate(generated_labels)


def evaluate_classifier(classifier, test_signals, test_labels, class_names, batch_size, device):
    '''Evaluate a classifier on a set of real ECG signals'''
    dataset = TensorDataset(
        torch.from_numpy(test_signals).float().unsqueeze(1),
        torch.from_numpy(test_labels).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    classifier.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for signals, labels in loader:
            logits = classifier(signals.to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            targets.extend(labels.numpy())

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
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


def train_classifier(train_signals, train_labels, validation_signals, validation_labels, num_classes, class_names, epochs, batch_size, learning_rate, patience, seed, device):
    '''Train a classifier and keep the checkpoint with the best validation macro F1'''
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
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_state = None
    best_validation_f1 = -np.inf
    epochs_without_improvement = 0
    history = {"train_loss": [], "validation_macro_f1": []}

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            loss = F.cross_entropy(classifier(signals), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        validation_results = evaluate_classifier(classifier, validation_signals, validation_labels, class_names, batch_size, device)
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


def plot_classifier_history(history, title, output_path):
    '''Save the classifier training curves'''
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
    '''Save a normalized confusion matrix and per-class metrics'''
    targets = results["targets"]
    predictions = results["predictions"]

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        normalize="true",
    )
    fig, axis = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", values_format=".2f", colorbar=False)
    axis.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200, bbox_inches="tight")
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
    per_class.to_csv(output_dir / f"{prefix}_per_class_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Train-on-Synthetic-Test-on-Real experiment"
    )
    parser.add_argument("--data-root", default="./processed_data")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--results-root", default="./test_runs")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--real-samples-per-class", type=int, default=300)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-epochs", type=int, default=80)
    parser.add_argument("--classifier-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_name = Path(args.checkpoint_path).stem
    output_dir = Path(args.results_root) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_info = load_preprocessing_metadata(args.data_root)
    class_names = preprocessing_info["class_names"]
    num_classes = preprocessing_info["num_classes"]
    train_data = np.load(Path(args.data_root) / "train.npz")
    validation_data = np.load(Path(args.data_root) / "validation.npz")
    test_data = np.load(Path(args.data_root) / "test.npz")

    vae, _ = load_vae_checkpoint(
        checkpoint_path=args.checkpoint_path,
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        device=device,
    )

    real_train_signals, real_train_labels = select_balanced_real_training_set(
        train_data,
        num_classes,
        args.real_samples_per_class,
        args.seed,
    )

    synthetic_train_signals, synthetic_train_labels = generate_balanced_synthetic_training_set(
        vae,
        args.embedding_dim,
        num_classes,
        args.synthetic_samples_per_class,
        device,
    )

    print("\nTraining classifier on real ECGs...")
    real_classifier, real_history = train_classifier(
        real_train_signals,
        real_train_labels,
        validation_data["signals"],
        validation_data["labels"],
        num_classes,
        class_names,
        args.classifier_epochs,
        args.batch_size,
        args.learning_rate,
        args.classifier_patience,
        args.seed,
        device,
    )
    trtr_results = evaluate_classifier(
        real_classifier,
        test_data["signals"],
        test_data["labels"],
        class_names,
        args.batch_size,
        device,
    )

    print("\nTraining classifier on synthetic ECGs...")
    synthetic_classifier, synthetic_history = train_classifier(
        synthetic_train_signals,
        synthetic_train_labels,
        validation_data["signals"],
        validation_data["labels"],
        num_classes,
        class_names,
        args.classifier_epochs,
        args.batch_size,
        args.learning_rate,
        args.classifier_patience,
        args.seed,
        device,
    )
    tstr_results = evaluate_classifier(
        synthetic_classifier,
        test_data["signals"],
        test_data["labels"],
        class_names,
        args.batch_size,
        device,
    )

    plot_classifier_history(real_history, "TRTR Classifier", output_dir / "trtr_training_history.png")
    plot_classifier_history(synthetic_history, "TSTR Classifier", output_dir / "tstr_training_history.png")
    save_classifier_diagnostics(
        trtr_results,
        class_names,
        "TRTR: Train on Real, Test on Real",
        output_dir,
        "trtr",
    )
    save_classifier_diagnostics(
        tstr_results,
        class_names,
        "TSTR: Train on Synthetic, Test on Real",
        output_dir,
        "tstr",
    )

    results = {
        "real_samples_per_class": args.real_samples_per_class,
        "synthetic_samples_per_class": args.synthetic_samples_per_class,
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
    with open(output_dir / "tstr_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nTSTR results")
    print(f"TRTR accuracy: {trtr_results['accuracy']:.4f}")
    print(f"TRTR macro F1: {trtr_results['macro_f1']:.4f}")
    print(f"TSTR accuracy: {tstr_results['accuracy']:.4f}")
    print(f"TSTR macro F1: {tstr_results['macro_f1']:.4f}")
    print(f"\nSaved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()