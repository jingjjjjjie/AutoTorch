'''
Simplified evaluation pipeline:
1. Preprocess data -> get combined batch DataFrame
2. Run inference -> add prediction columns
3. Store predictions to CSV
'''
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.idfraud.preprocessing import preprocess_csv, map_path_to_source
from data.idfraud.transforms import get_transform
from data.idfraud.dataset import IDFraudTorchDataset
from models import build_classifier_model, load_weights_from_checkpoint
from models.dino import load_dino_model


def get_batch_dataframe(batch_list, image_type='ori'):
    """
    Preprocess and combine all batch CSVs into a single DataFrame.

    Args:
        batch_list: List of batch CSV paths (relative to MNT3PATH)
        image_type: One of 'ori', 'crop', 'corner'

    Returns:
        pd.DataFrame: Combined DataFrame with resolved paths
    """
    print("Loading and combining batch CSVs...")
    main_data, missing_batches = preprocess_csv(
        image_type=image_type,
        batch_list=batch_list,
        training_mode=False
    )

    if missing_batches:
        print(f"Warning: Missing batches: {missing_batches}")

    print("Mapping paths to source...")
    df, missing_paths = map_path_to_source(main_data, training_mode=False)

    if missing_paths:
        print(f"Warning: {len(missing_paths)} paths not found")
        # Remove rows with missing paths
        df = df[df['path'].notna()].reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    return df


def run_inference(model, df, cfg, device='cuda', batch_size=32, threshold=0.5):
    """
    Run inference on a DataFrame and add prediction columns.

    Args:
        model: Loaded PyTorch model
        df: DataFrame with 'path' column pointing to images
        cfg: Config dict with transform settings
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        threshold: Classification threshold

    Returns:
        pd.DataFrame: Original df with 'pred_prob' and 'pred_label' columns added
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    transform = get_transform(cfg)
    dataset = IDFraudTorchDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    print(f"Running inference on {len(df)} samples...")
    with torch.inference_mode():
        for X, _ in tqdm(dataloader, desc="Inference"):
            X = X.to(device)
            logits = model(X).squeeze(dim=1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())

    # Add prediction columns
    df = df.copy()
    df['pred_prob'] = all_probs
    df['pred_label'] = [1 if p > threshold else 0 for p in all_probs]

    return df


def save_predictions(df, output_path):
    """Save DataFrame with predictions to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def run_evaluation_pipeline(cfg, checkpoint_path, batch_list, output_path,
                            device='cuda', image_type='ori', batch_size=32, threshold=0.5):
    """
    Complete evaluation pipeline:
    1. Preprocess data -> combined DataFrame
    2. Load model and run inference
    3. Save predictions to CSV

    Args:
        cfg: Config dict
        checkpoint_path: Path to model checkpoint
        batch_list: List of batch CSV paths
        output_path: Path to save predictions CSV
        device: 'cuda' or 'cpu'
        image_type: One of 'ori', 'crop', 'corner'
        batch_size: Batch size for inference
        threshold: Classification threshold

    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Step 1: Get combined batch DataFrame
    print("\n=== Step 1: Preprocessing Data ===")
    df = get_batch_dataframe(batch_list, image_type=image_type)

    # Step 2: Load model
    print("\n=== Step 2: Loading Model ===")
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    backbone = load_dino_model(cfg)
    model = build_classifier_model(cfg, device_obj, backbone)
    model = load_weights_from_checkpoint(model, checkpoint_path, device_obj)
    print(f"Model loaded from: {checkpoint_path}")

    # Step 3: Run inference
    print("\n=== Step 3: Running Inference ===")
    df = run_inference(model, df, cfg, device=device, batch_size=batch_size, threshold=threshold)

    # Step 4: Save predictions
    print("\n=== Step 4: Saving Predictions ===")
    save_predictions(df, output_path)

    # Print summary
    if 'label' in df.columns:
        correct = (df['pred_label'] == df['label']).sum()
        accuracy = correct / len(df)
        print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(df)})")

    return df


# Example usage
if __name__ == "__main__":
    import yaml

    # Load config
    config_path = "path/to/config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Define batch list
    batch_list = [
        "batch_datacollection_20230214_none_recapture/batch_datacollection_20230214_recapture_train_set.csv",
        "batch_production_20240206_none_1000/index_annotation_mykadfront_annotated_train_set.csv",
    ]

    # Run pipeline
    df = run_evaluation_pipeline(
        cfg=cfg,
        checkpoint_path="path/to/checkpoint.pt",
        batch_list=batch_list,
        output_path="predictions.csv",
        device='cuda',
        image_type='ori',
        batch_size=32,
        threshold=0.5
    ) 
