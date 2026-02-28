'''
Training curve plotting utilities.
'''
import os
import matplotlib.pyplot as plt
from typing import Dict, List


def plot_results(results: Dict[str, List], save_dir: str = None):
    """Plot training curves and save as separate files."""
    epochs = range(1, len(results["train_loss"]) + 1)

    def save_plot(name):
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150)
        plt.close()

    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, results["train_loss"], label="Train")
    plt.plot(epochs, results["test_loss"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()
    save_plot("loss")

    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, results["train_acc"], label="Train")
    plt.plot(epochs, results["test_acc"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend()
    save_plot("accuracy")

    # APCER
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, results["train_apcer"], label="Train")
    plt.plot(epochs, results["test_apcer"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("APCER"); plt.title("APCER (target: 0.03)"); plt.legend()
    plt.axhline(y=0.03, color='r', linestyle='--', alpha=0.5, label='Target')
    save_plot("apcer")

    # BPCER
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, results["train_bpcer"], label="Train")
    plt.plot(epochs, results["test_bpcer"], label="Val")
    plt.xlabel("Epoch"); plt.ylabel("BPCER"); plt.title("BPCER (target: 0.05)"); plt.legend()
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Target')
    save_plot("bpcer")

    # Learning Rate
    if results.get("lr"):
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, results["lr"])
        plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.title("Learning Rate Schedule")
        save_plot("lr")
