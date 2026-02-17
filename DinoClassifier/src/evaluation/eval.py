"""
Minimal evaluation script for DinoClassifier.
Outputs info.json compatible with automl_platform leaderboard.
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
    apcer = fn / (tp + fn) if (tp + fn) else 0  # Attack Presentation Classification Error Rate
    bpcer = fp / (tn + fp) if (tn + fp) else 0  # Bona Fide Presentation Classification Error Rate
    acer = (apcer + bpcer) / 2
    return accuracy, apcer, bpcer, acer


def evaluate(model, dataloader, device, threshold=0.5):
    """Run evaluation and return metrics."""
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Evaluating"):
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

    metrics = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
    }

    return metrics, all_preds, all_labels, all_probs


def load_model(cfg, checkpoint_path, device):
    """Load model from checkpoint."""
    backbone = build_dino_backbone(cfg)
    head = build_head(cfg)
    model = CustomClassifierModel(
        backbone=backbone,
        head=head,
        freeze_backbone=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model.to(device)


def save_results(output_dir, dataset_name, metrics, df=None):
    """Save info.json (and optionally CSV) in leaderboard-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    # Save info.json in expected format
    info = {
        "metrics": {
            "threshold_0.5": metrics
        }
    }
    info_path = os.path.join(output_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"Saved: {info_path}")

    # Optionally save predictions CSV
    if df is not None:
        csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DinoClassifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--eval_csv", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config and model
    cfg = load_config(args.config)
    model = load_model(cfg, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load eval data
    transform = get_transform(cfg)
    eval_df = pd.read_csv(args.eval_csv)
    eval_dataset = CSVTorchDataset(eval_df, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Loaded {len(eval_dataset)} samples from {args.eval_csv}")

    # Run evaluation
    metrics, preds, labels, probs = evaluate(model, eval_loader, device, args.threshold)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"APCER:    {metrics['apcer']:.4f}")
    print(f"BPCER:    {metrics['bpcer']:.4f}")
    print(f"ACER:     {metrics['acer']:.4f}")
    print(f"TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
    print("=" * 50)

    # Save results
    dataset_name = os.path.splitext(os.path.basename(args.eval_csv))[0]
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    result_dir = os.path.join(args.output_dir, checkpoint_name, dataset_name)

    # Add predictions to dataframe
    eval_df["prediction"] = preds
    eval_df["probability"] = probs

    save_results(result_dir, dataset_name, metrics, eval_df)
    print(f"\nResults saved to: {result_dir}")


if __name__ == "__main__":
    main()
