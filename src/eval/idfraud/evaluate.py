"""Run evaluation on test batches after training."""
import os
import re
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.idfraud.preprocessing import preprocess_csv, map_path_to_source
from data.idfraud.transforms import build_transform
from data.idfraud.dataset import IDFraudTorchDataset
from models import build_model, load_weights_from_checkpoint
from utils.device import is_main_process

PRED_DF_OUTPUT_PATH = 'eval_predictions.csv'

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


def run_evaluation(cfg, device='cuda'):
    
    if not is_main_process():
        return None

    # Load config
    save_dir = cfg.experiment.save_dir
    save_name = cfg.experiment.save_name
    run_dir = os.path.join(save_dir, save_name)
    eval_batches = cfg.data.eval_batches
    image_type = cfg.data.image_type
    batch_size = cfg.training.batch_size
    pin_memory = cfg.dataloader.pin_memory
    num_workers = cfg.dataloader.num_workers
    prefetch_factor = cfg.dataloader.prefetch_factor
    non_blocking= cfg.dataloader.non_blocking
    persistent_workers = cfg.dataloader.persistent_workers
    backbone_name = cfg.model.backbone_name
    output_type = cfg.model.output_type  # 'logits' or 'probs'

    # Find checkpoints
    checkpoint_folder = os.path.join(run_dir, 'checkpoints')
    checkpoints = find_checkpoints(checkpoint_folder)

    # Preprocess data
    evaluation_df = preprocess_csv(image_type=image_type, batch_list=eval_batches, training_mode=False)
    evaluation_df = map_path_to_source(evaluation_df, training_mode=False)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    transform = build_transform(
        image_size=cfg.transform.image_size,
        normalize_mean=tuple(cfg.transform.normalize_mean),
        normalize_std=tuple(cfg.transform.normalize_std),
        version=cfg.transform.get('version', 'v1'))
    dataset = IDFraudTorchDataset(evaluation_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    output_path = os.path.join(run_dir, PRED_DF_OUTPUT_PATH)

    # Run inference for each checkpoint
    for epoch_num, ckpt_path in checkpoints:
        model = build_model(
            model_name=backbone_name,
            device=device,
            task=cfg.model.get('task', 'classification'),
            head_type=cfg.model.head_type,
            freeze_backbone=cfg.model.freeze_backbone)
        model = load_weights_from_checkpoint(model, ckpt_path, device)
        model.eval()

        all_probs = []
        with torch.inference_mode():
            for X, _ in tqdm(dataloader, desc=f"Inference ckpt{epoch_num}"):
                X = X.to(device, non_blocking=non_blocking)
                # Forward pass
                output = model(X)
                # Remove channel dimension
                output = output.squeeze(1)
                # Convert to probabilities (only if model outputs logits)
                probs = torch.sigmoid(output) if output_type == 'logits' else output
                # Move to CPU and convert to Python list
                probs = probs.cpu().tolist()
                # Store results
                all_probs.extend(probs)

        col = f'pred_prob_ckpt{epoch_num}'
        evaluation_df[col] = all_probs

        # Save after each checkpoint
        evaluation_df.to_csv(output_path, index=False)

    return evaluation_df
