"""
Run inference on the eval set using the best checkpoint and write eval_predictions.csv.

Usage:
    python src/evaluate.py --run-dir /path/to/experiment
"""
import sys
import argparse
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from data.idfraud.transforms import build_transform
from data.idfraud.dataset import IDFraudTorchDataset
from models import build_model, load_weights_from_checkpoint
from utils.device import is_main_process

PRED_CSV      = 'eval_predictions.csv'
SELECTION_CSV = 'checkpoint_selection.csv'


def find_best_checkpoint(run_dir):
    """Return (epoch, ckpt_path) — rank 1 row from checkpoint_selection.csv."""
    epoch = int(pd.read_csv(os.path.join(run_dir, SELECTION_CSV)).iloc[0]['epoch'])
    ckpt_path = os.path.join(run_dir, 'checkpoints', f'epoch_{epoch}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return epoch, ckpt_path


def run_evaluation(cfg):
    if not is_main_process():
        return

    epoch, ckpt_path = find_best_checkpoint(cfg.run_dir)
    print(f"Evaluating best checkpoint: epoch {epoch} ({ckpt_path})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(cfg.data.eval_csv)
    transform = build_transform(
        image_size=cfg.transform.image_size,
        normalize_mean=tuple(cfg.transform.normalize_mean),
        normalize_std=tuple(cfg.transform.normalize_std),
        version=cfg.transform.get('version', 'v1'))
    dataloader = DataLoader(
        IDFraudTorchDataset(df, transform=transform),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor)

    model = build_model(
        model_name=cfg.model.backbone_name,
        device=device,
        task=cfg.model.get('task', 'classification'),
        head_type=cfg.model.head_type,
        freeze_backbone=cfg.model.freeze_backbone)
    model = load_weights_from_checkpoint(model, ckpt_path, device)
    model.eval()

    all_probs = []
    with torch.inference_mode():
        for X, _ in tqdm(dataloader, desc=f"Inference ckpt{epoch}"):
            output = model(X.to(device, non_blocking=cfg.dataloader.non_blocking)).squeeze(1)
            probs = torch.sigmoid(output) if cfg.model.output_type == 'logits' else output
            all_probs.extend(probs.cpu().tolist())

    pred_col = f'pred_prob_ckpt{epoch}'
    df[pred_col] = all_probs
    out_path = os.path.join(cfg.run_dir, PRED_CSV)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  (pred col: {pred_col})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on eval set.")
    parser.add_argument("--run-dir", required=True, help="Experiment directory (must contain config.yaml).")
    args = parser.parse_args()
    cfg = OmegaConf.load(os.path.join(args.run_dir, 'config.yaml'))
    cfg.run_dir = args.run_dir
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
