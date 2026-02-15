"""
Post-training utilities: saving results and plotting.
"""
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from utils.model import save_model


def save_run(cfg, results, model, df_train, df_val, elapsed):
    """Save all training artifacts: config, data splits, logs, plots, model."""
    run_name = cfg['model']['save_name'].replace('.pth', '')
    run_dir = os.path.join(cfg['model']['save_dir'], run_name)
    plots_dir = os.path.join(run_dir, 'plots')
    dataframe_dir = os.path.join(run_dir, 'dataframes')

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(dataframe_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Save data splits
    df_train.to_csv(os.path.join(dataframe_dir, 'train_data.csv'), index=False)
    df_val.to_csv(os.path.join(dataframe_dir, 'val_data.csv'), index=False)

    # Save training logs
    pd.DataFrame(results).to_csv(os.path.join(run_dir, 'log.csv'), index_label='epoch')

    # Plot training results
    plot_results(results, save_dir=plots_dir)

    # Save model (unwrap DDP with .module)
    save_model(model=model.module, target_dir=run_dir, model_name=cfg['model']['save_name'])

    # Print total time
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    print("=" * 40)
    print(f"Total time: {hrs:02d}:{mins:02d}:{secs:02d}")
    print("=" * 40)


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
