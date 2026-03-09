"""Run inference on batch CSVs and save predictions."""
import os
import re
import sys
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.idfraud.preprocessing import preprocess_csv, map_path_to_source
from data.idfraud.transforms import get_transform
from data.idfraud.dataset import IDFraudTorchDataset
from models import build_classifier_model, load_weights_from_checkpoint
from models.dino import load_dino_model
from eval.idfraud.calculate import calculate_metrics, print_metrics


def find_checkpoints(checkpoint_folder):
    """Find all epoch_x.pt files in folder and return sorted list of (epoch_num, path)."""
    checkpoints = []
    pattern = re.compile(r'^epoch_(\d+)\.pt$')
    for fname in os.listdir(checkpoint_folder):
        match = pattern.match(fname)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, os.path.join(checkpoint_folder, fname)))
    return sorted(checkpoints, key=lambda x: x[0])


def run_evaluation(cfg, checkpoint_folder, output_path, batch_list=None, num_workers=4, pin_memory=True,
                   device='cuda', image_type='crop', batch_size=32, prefetch_factor=4, epochs=None):
    """
    Run inference on batches using checkpoints in folder.

    batch_list: if None, reads from cfg['eval_batches'].
    epochs: list of epoch numbers to evaluate, e.g. [1, 2] or [5]. If None, evaluates all.
    Returns DataFrame with pred_prob_ckpt{x} columns for each checkpoint.
    """
    if batch_list is None:
        batch_list = cfg['eval_batches']

    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_folder)
    if not checkpoints:
        raise ValueError(f"No epoch_x.pt files found in {checkpoint_folder}")

    # Filter to specific epochs if requested
    if epochs is not None:
        checkpoints = [(e, p) for e, p in checkpoints if e in epochs]
        if not checkpoints:
            raise ValueError(f"No checkpoints found for epochs {epochs}")
    print(f"Found {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}")

    # Preprocess data
    print("Loading data...")
    df = map_path_to_source(
        preprocess_csv(image_type=image_type, batch_list=batch_list, training_mode=False),
        training_mode=False
    )
    print(f"Samples: {len(df)}")

    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    transform = get_transform(cfg)
    dataset = IDFraudTorchDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )


    # Load backbone once
    print("Loading backbone...")
    backbone = load_dino_model(cfg)

    df = df.copy()

    # Run inference for each checkpoint
    for epoch_num, ckpt_path in checkpoints:
        print(f"\nEvaluating checkpoint epoch_{epoch_num}...")
        model = build_classifier_model(cfg, device, backbone)
        model = load_weights_from_checkpoint(model, ckpt_path, device)
        model.eval()

        all_probs = []
        with torch.inference_mode():
            for X, _ in tqdm(dataloader, desc=f"Inference ckpt{epoch_num}"):
                logits = model(X.to(device)).squeeze(1)
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())

        col = f'pred_prob_ckpt{epoch_num}'
        df[col] = all_probs

        # Save after each checkpoint
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

        # Print metrics
        print(f"Epoch {epoch_num} metrics:")
        print_metrics(calculate_metrics(df, prob_col=col))

    return df


if __name__ == "__main__":

    RUNS_DIR      = "/home/jingjie/AutoTorch/runs"

    # change this two each eval
    EXP_NAME      = "Exp2_dinov3_convnext_large_v1_512_ori"
    EVAL_CONFIG   = "/home/jingjie/AutoTorch/configs/eval_ori_dataset_21v1.yaml"

    exp_dir = os.path.join(RUNS_DIR, EXP_NAME)

    with open(os.path.join(exp_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    with open(EVAL_CONFIG) as f:
        eval_cfg = yaml.safe_load(f)

    df = run_evaluation(
        cfg=cfg,
        checkpoint_folder=os.path.join(exp_dir, "checkpoints"),
        batch_list=eval_cfg['eval_batches'],
        output_path=os.path.join(exp_dir, "predictions_full.csv"),
        batch_size=40, # best was 40
        num_workers=24, # best was 24
        pin_memory=True,
        prefetch_factor=10, # best was 10
        image_type=eval_cfg['image_type'],
        device='cuda:0',
    )
