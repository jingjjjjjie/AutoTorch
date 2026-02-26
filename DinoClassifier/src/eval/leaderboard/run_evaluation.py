'''
Evaluation pipeline using existing functions.
'''
import os
import sys
import glob
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.preprocessing import preprocess_csv, map_path_to_source
from data.transforms import get_transform
from data.idfraud_torch_dataset import IDFraudTorchDataset
from models import build_classifier_model, load_weights_from_checkpoint
from eval.classification import run_eval_classification, compute_binary_metrics
from eval.leaderboard.generate_eval_structure import generate_eval_structure, update_batch_metrics


def get_threshold_metrics_dataframe(probs, labels):
    """Compute metrics at thresholds 0.0 to 0.95 and returns a dataframe"""
    thresholds = [round(t * 0.05, 2) for t in range(20)]
    results = []

    for thresh in thresholds:
        tp, tn, fp, fn = 0, 0, 0, 0
        for prob, label in zip(probs, labels):
            pred = 1 if prob > thresh else 0
            if pred == 1 and label == 1: tp += 1
            elif pred == 0 and label == 0: tn += 1
            elif pred == 1 and label == 0: fp += 1
            else: fn += 1

        acc, apcer, bpcer = compute_binary_metrics(tp, tn, fp, fn)
        results.append({'threshold': thresh, 'acc': acc, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'apcer': apcer, 'bpcer': bpcer})

    return pd.DataFrame(results)


def run_full_evaluation(cfg, checkpoint_path, batch_list, run_path, device='cuda', image_type='ori', batch_size=16, threshold=0.5):
    """Run full evaluation using existing functions."""
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # 1. Load model
    print("\n1. Loading model...")
    model = build_classifier_model(cfg, device, load_weights=False)
    model = load_weights_from_checkpoint(model, checkpoint_path, device)
    model.eval()

    # 2. Load data using preprocess_csv and map_path_to_source
    print("\n2. Loading data...")
    main_data, missing_batches = preprocess_csv(image_type=image_type, batch_list=batch_list, training_mode=False)
    if missing_batches:
        raise FileNotFoundError(f"CSV batch not found: {missing_batches}")

    df_fixed, missing_paths = map_path_to_source(main_data, training_mode=False)
    if missing_paths:
        errors_path = os.path.join(run_path, 'eval', 'path_errors.csv')
        os.makedirs(os.path.dirname(errors_path), exist_ok=True)
        pd.DataFrame({'path': missing_paths}).to_csv(errors_path, index=False)
        raise FileNotFoundError(f"{len(missing_paths)} paths not found, saved to {errors_path}")

    print(f"   {len(df_fixed)} samples loaded")

    # 3. Get transform using existing get_transform
    transform = get_transform(cfg)

    # 4. Generate eval structure using existing function
    # checkpoint_path: runs/Ex2_vits16_226test/checkpoints/epoch_10.pt -> checkpoint_name: epoch_10 
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '').replace('.pth', '')
    
    eval_config = {"eval_batch_size": batch_size, "bin_threshold": threshold, "image_type": image_type}

    print("\n3. Generating eval structure...")
    eval_path = generate_eval_structure(run_path, [checkpoint_name], df_fixed)

    # 5. Evaluate each batch
    print("\n4. Running evaluation...")
    for batch in df_fixed['batch_directory'].unique(): # for each batch in the list
        print(f"\n   Batch: {batch}")
        batch_df = df_fixed[df_fixed['batch_directory'] == batch].copy()

        # Create dataloader using existing IDFraudTorchDataset
        dataset = IDFraudTorchDataset(batch_df, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Run eval using existing run_eval_classification
        metrics, preds, probs = run_eval_classification(model, dataloader, device, threshold)

        # Compute threshold metrics
        labels = batch_df['label'].tolist()
        threshold_df = get_threshold_metrics_dataframe(probs, labels)

        # Format metrics for info.json
        metrics_dict = {"test": {**metrics, "threshold": threshold}}

        # Save metrics using update_batch_metrics
        batch_folder = batch.replace('/', '_').replace('\\', '_')
        batch_path = os.path.join(eval_path, checkpoint_name, batch_folder)
        update_batch_metrics(batch_path, metrics_dict, threshold_df, eval_config, batch_list, checkpoint_name)

        # Save predictions to CSV
        batch_df['pred_prob'] = probs
        batch_df['pred_label'] = preds
        batch_df.to_csv(os.path.join(batch_path, f'{batch_folder}.csv'), index=False)

        apcer_str = f"{metrics['apcer']:.4f}" if metrics['apcer'] != -1 else "N/A"
        bpcer_str = f"{metrics['bpcer']:.4f}" if metrics['bpcer'] != -1 else "N/A"
        print(f"   Acc: {metrics['accuracy']:.4f}, APCER: {apcer_str}, BPCER: {bpcer_str}")

    print(f"\n5. Done! Results: {eval_path}")
    return eval_path


def run_evaluation_from_folder(run_path, batch_list, checkpoints='all', device='cuda', image_type='ori', batch_size=16, threshold=0.5):
    """
    Run evaluation for all checkpoints in a run folder.

    Args:
        run_path: Path to run folder (e.g., runs/Ex2_vits16_226test)
        batch_list: List of batch CSV paths to evaluate
        checkpoints: 'all' for all checkpoints, or list of checkpoint names like ['epoch_10', 'epoch_20']
        device: 'cuda' or 'cpu'
        image_type: Image type to use ('ori', 'crop', or 'corner')
        batch_size: Batch size for evaluation
        threshold: Classification threshold
    """
    # 1. Read config from run folder
    config_path = os.path.join(run_path, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")
    print(f"  Model: {cfg['model']['backbone_name']}")
    print(f"  Image size: {cfg['transform']['image_size']}")

    # 2. Find checkpoints
    checkpoint_dir = os.path.join(run_path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoints folder not found: {checkpoint_dir}")

    if checkpoints == 'all':
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pt')))
        # Sort by epoch number
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(os.path.basename(x).replace('epoch_', '').replace('.pt', '')))
    else:
        checkpoint_files = [os.path.join(checkpoint_dir, f"{name}.pt") for name in checkpoints]
        # Verify they exist
        for cp in checkpoint_files:
            if not os.path.exists(cp):
                raise FileNotFoundError(f"Checkpoint not found: {cp}")

    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")

    # 3. Run evaluation for each checkpoint
    results = {}
    for checkpoint_path in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '').replace('.pth', '')
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_name}")
        print(f"{'='*60}")

        eval_path = run_full_evaluation(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            batch_list=batch_list,
            run_path=run_path,
            device=device,
            image_type=image_type,
            batch_size=batch_size,
            threshold=threshold
        )
        results[checkpoint_name] = eval_path

    print(f"\n{'='*60}")
    print(f"All evaluations complete! Results in: {os.path.join(run_path, 'eval')}")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    # Just specify run folder and batches to evaluate
    run_path = "/home/jingjie/DinoFT/DinoClassifier/runs/Ex2_vits16_226test"
    batch_list = [
        "batch_issue_20230322_none_recapture_colorprint/index_annotation_mykadfront.csv",
        "batch_issue_20240704_snt_both_colorghostwhitebg/index_annotation_mykadfront.csv",
    ]

    try:
        # Evaluate all checkpoints
        # results = run_evaluation_from_folder(run_path, batch_list, checkpoints='all')

        # Or evaluate specific checkpoints
        results = run_evaluation_from_folder(
            run_path=run_path,
            batch_list=batch_list,
            checkpoints='all',  # or 'all'
            image_type='ori',
        )
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
