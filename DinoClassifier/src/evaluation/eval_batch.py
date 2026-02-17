"""
Batch evaluation script - evaluate multiple checkpoints on multiple datasets.
Outputs results compatible with automl_platform leaderboard.

Usage:
    python eval_batch.py --config config.yaml --checkpoint_dir runs/Experiment1/checkpoints --eval_csvs data1.csv data2.csv --output_dir infer_results
"""
import os
import json
import argparse
import torch
import pandas as pd
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import CSVTorchDataset
from data.transforms import get_transform
from models.backbone import build_dino_backbone
from models.head import build_head
from models.classifier import CustomClassifierModel
from utils.config import load_config

def compute_metrics(tp, tn, fp, fn):
    """Compute accuracy, apcer, bpcer, acer from counts."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    apcer = fn / (tp + fn) if (tp + fn) else 0
    bpcer = fp / (tn + fp) if (tn + fp) else 0
    acer = (apcer + bpcer) / 2
    return accuracy, apcer, bpcer, acer

def evaluate(model, dataloader, device, threshold=0.5):
    """Run evaluation and return metrics."""
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(dim=1)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()

            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy, apcer, bpcer, acer = compute_metrics(tp, tn, fp, fn)

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy, "apcer": apcer, "bpcer": bpcer, "acer": acer,
    }, all_preds, all_probs


def load_model(cfg, checkpoint_path, device):
    """Load model from checkpoint."""
    backbone = build_dino_backbone(cfg)
    head = build_head(cfg)
    model = CustomClassifierModel(backbone=backbone, head=head, freeze_backbone=False)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return model.to(device)


def save_results(output_dir, dataset_name, metrics, df=None):
    """Save info.json in leaderboard-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    info = {"metrics": {"threshold_0.5": metrics}}
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    if df is not None:
        df.to_csv(os.path.join(output_dir, f"{dataset_name}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate DinoClassifier checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory containing checkpoints")
    parser.add_argument("--checkpoints", nargs="+", help="List of specific checkpoint paths")
    parser.add_argument("--eval_csvs", nargs="+", required=True, help="Evaluation CSV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config)
    transform = get_transform(cfg)

    # Get checkpoints
    if args.checkpoints:
        checkpoints = args.checkpoints
    elif args.checkpoint_dir:
        checkpoints = sorted(glob(os.path.join(args.checkpoint_dir, "*.pt")))
    else:
        raise ValueError("Provide --checkpoint_dir or --checkpoints")

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Evaluating on {len(args.eval_csvs)} datasets")

    # Results summary
    summary = []

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"{'='*60}")

        model = load_model(cfg, ckpt_path, device)

        for csv_path in args.eval_csvs:
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
            print(f"  Dataset: {dataset_name}")

            # Load data
            df = pd.read_csv(csv_path)
            dataset = CSVTorchDataset(df, transform=transform)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            # Evaluate
            metrics, preds, probs = evaluate(model, loader, device)

            # Save
            result_dir = os.path.join(args.output_dir, ckpt_name, dataset_name)
            df["prediction"] = preds
            df["probability"] = probs
            save_results(result_dir, dataset_name, metrics, df)

            # Print
            print(f"    Acc: {metrics['accuracy']:.4f} | APCER: {metrics['apcer']:.4f} | BPCER: {metrics['bpcer']:.4f} | ACER: {metrics['acer']:.4f}")

            summary.append({
                "checkpoint": ckpt_name,
                "dataset": dataset_name,
                **metrics
            })

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.output_dir, "eval_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
