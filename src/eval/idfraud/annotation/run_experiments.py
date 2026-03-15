"""
Automated experiment runner for ID fraud evaluation.

Reads experiments.csv and runs inference for any missing prediction columns.
"""
import sys
import os
import shutil
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/jingjie/AutoTorch/src')

from data.idfraud.dataset import IDFraudTorchDataset
from data.idfraud.transforms import get_transform
from models import build_model, load_weights_from_checkpoint
from omegaconf import OmegaConf

# Paths
ANNOTATION_DIR = os.path.dirname(os.path.abspath(__file__))
DATAFRAMES_DIR = os.path.join(ANNOTATION_DIR, 'dataframes')
EXPERIMENTS_CSV = os.path.join(ANNOTATION_DIR, 'experiments.csv')
SOURCE_CSV = os.path.join(DATAFRAMES_DIR, 'processed_recapture_tamper_feb.csv')
PROCESSED_CSV = os.path.join(DATAFRAMES_DIR, 'processed_recapture_tamper_feb_processed.csv')


def run_inference(df,
                  path_col,
                  gt_col,
                  exp_dir,
                  ckpt_num,
                  cfg,
                  transform,
                  device,
                  batch_size=32):
    """Run inference using specified path column and checkpoint."""
    df_temp = df.copy()
    df_temp['path'] = df_temp[path_col]

    dataset = IDFraudTorchDataset(df_temp, transform=transform, gt_label=gt_col, img_path=path_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    ckpt_path = os.path.join(exp_dir, "checkpoints", f"epoch_{ckpt_num}.pt")
    model = build_model(
        model_name=cfg.model.backbone_name,
        device=device,
        head_type=cfg.model.head_type
    )
    model = load_weights_from_checkpoint(model, ckpt_path, device)
    model.eval()

    output_type = cfg.model.get('output_type', 'logits')
    all_probs = []
    with torch.inference_mode():
        for X, _ in tqdm(dataloader, desc=f"Inference {os.path.basename(exp_dir)}", total=len(dataloader), unit="batch"):
            output = model(X.to(device)).squeeze(1)
            probs = output if output_type == 'probs' else torch.sigmoid(output)
            all_probs.extend(probs.cpu().tolist())

    return all_probs


def process_experiment(df, exp_row, device):
    """Process a single experiment row, adding predictions if missing."""
    exp_name = exp_row['experiment_name']
    ori_col = f"{exp_name}_pred_ori"
    crop_col = f"{exp_name}_pred_crop"

    # Check which columns need to be added
    need_ori = ori_col not in df.columns
    need_crop = crop_col not in df.columns

    if not need_ori and not need_crop:
        print(f"[{exp_name}] All prediction columns already exist, skipping.")
        return df

    # if the prediction columns is not found:
    # Check if experiment directories exist
    ori_dir = exp_row['ori_exp_dir']
    crop_dir = exp_row['crop_exp_dir']

    # logic to check if experiment directory is present
    for label, d in [('ORI', ori_dir), ('CROP', crop_dir)]:
        if not d or pd.isna(d):
            print(f"[{exp_name}] Missing experiment directories, skipping.")
            return df
        if not os.path.exists(d):
            print(f"[{exp_name}] {label} directory not found: {d}, skipping.")
            return df

    # Load config and transform (use ori config as reference)
    cfg = OmegaConf.load(os.path.join(ori_dir, "config.yaml"))
    transform = get_transform(
        image_size=cfg.transform.image_size,
        normalize_mean=cfg.transform.normalize_mean,
        normalize_std=cfg.transform.normalize_std
    )

    # Run inference for missing columns
    common = dict(df=df, gt_col='annot_is_idfraud', cfg=cfg, transform=transform, device=device)
    if need_ori:
        print(f"[{exp_name}] Running ORI inference...")
        df[ori_col] = run_inference(**common, path_col='absolute_ori_path', exp_dir=ori_dir, ckpt_num=int(exp_row['ori_best_ckpt']))
    if need_crop:
        print(f"[{exp_name}] Running CROP inference...")
        df[crop_col] = run_inference(**common, path_col='absolute_ocr_path', exp_dir=crop_dir, ckpt_num=int(exp_row['crop_best_ckpt']))

    # Add derived label columns
    for suffix, col in [('ori', ori_col), ('crop', crop_col)]:
        df[f"{exp_name}_{suffix}_label"] = (df[col] > 0.5).astype(int)
    df[f"{exp_name}_parallel_label"] = ((df[ori_col] + df[crop_col]) / 2 > 0.5).astype(int)

    print(f"[{exp_name}] Done!")
    return df


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Check/create processed CSV
    if not os.path.exists(PROCESSED_CSV):
        if os.path.exists(SOURCE_CSV):
            print(f"Copying {SOURCE_CSV} -> {PROCESSED_CSV}")
            shutil.copy(SOURCE_CSV, PROCESSED_CSV)
        else:
            raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    # Step 2: Load data
    df = pd.read_csv(PROCESSED_CSV)

    # Fix paths if needed
    for col in ['absolute_ori_path', 'absolute_ocr_path']:
        df[col] = df[col].str.replace('/routine_data/', '/routine_data_jj_feb/')

    print(f"Loaded {len(df)} rows")
    print(f"Existing columns: {list(df.columns)}")

    # Step 3: Load experiments
    experiments = pd.read_csv(EXPERIMENTS_CSV)
    print(f"\nFound {len(experiments)} experiments:")
    print(experiments)

    # Step 4: Process each experiment
    for _, exp_row in tqdm(experiments.iterrows(), total=len(experiments), desc="Experiments"):
        df = process_experiment(df, exp_row, device)
        # Save after each experiment (in case of crash)
        df.to_csv(PROCESSED_CSV, index=False)

    print(f"\nFinal columns: {list(df.columns)}")
    print(f"Saved to: {PROCESSED_CSV}")


if __name__ == "__main__":
    main()
