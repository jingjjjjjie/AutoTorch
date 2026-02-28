'''
Script to generate evaluation folder structure for model checkpoints.
Creates organized folders for storing inference results, sample images, and metrics.
'''
import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def generate_eval_structure(
    run_path: str,
    checkpoint_names: List[str],
    main_df: pd.DataFrame,
):
    """
    Generate evaluation folder structure with CSVs and placeholder metrics.

    Args:
        run_path: Path to the experiment folder (e.g., runs/Ex2_vits16_226test)
        checkpoint_names: List of checkpoint folder names
        main_df: DataFrame containing all evaluation data (must have 'batch' and 'path' columns)
    """

    # Create base eval structure
    # creates the eval directory inside the experiment folder
    eval_path = os.path.join(run_path, 'eval') 
    os.makedirs(eval_path, exist_ok=True)

    # loop through the checkpoints
    for checkpoint_name in checkpoint_names:
        checkpoint_path = os.path.join(eval_path, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Get unique batches from the dataframe
        unique_batches = main_df['batch_directory'].unique()

        for batch in unique_batches:
            # Create batch folder
            # sanitize folder name (replaces / with underscores)
            batch_folder_name = batch.replace('/', '_').replace('\\', '_')  
            batch_path = os.path.join(checkpoint_path, batch_folder_name)
            os.makedirs(batch_path, exist_ok=True)
            print(f"Created: {batch_path}")

    print(f"\nEval structure created at: {eval_path}")
    return eval_path


def create_info_json(
    batch_list: List[str],
    eval_config: dict,
    checkpoint_name: str,
    metrics: Optional[dict] = None
) -> dict:
    """Create the info.json structure."""

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d-%H%M")

    info = {
        "datetime": datetime_str,
        "eval": {
            "eval_data_source": {
                "eval": batch_list
            },
            "model_checkpoint": checkpoint_name,
            "eval_batch_size": eval_config.get('eval_batch_size', 16),
            "preprocess_type": eval_config.get('preprocess_type', 'imagenet-default'),
            "bin_threshold": eval_config.get('bin_threshold', 0.5),
            "csv_image_column": eval_config.get('csv_image_column', 'ori-path')
        },
        "metrics": metrics if metrics else {
            "test": "INCOMPLETE"
        }
    }

    return info


def update_batch_metrics(
    batch_path: str,
    metrics: dict,
    threshold_metrics: pd.DataFrame,
    eval_config: dict,
    batch_list: List[str],
    checkpoint_name: str
):
    """
    Save metrics for a specific batch folder after inference.

    Args:
        batch_path: Path to the batch folder
        metrics: Dictionary with metrics for info.json
        threshold_metrics: DataFrame with threshold-wise metrics
        eval_config: Evaluation configuration
        batch_list: List of batch paths used
        checkpoint_name: Name of the checkpoint
    """
    # Create info.json with real metrics
    info = create_info_json(
        batch_list=batch_list,
        eval_config=eval_config,
        checkpoint_name=checkpoint_name,
        metrics=metrics
    )
    info_path = os.path.join(batch_path, 'info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent='\t')

    # Save threshold_metrics.csv
    threshold_path = os.path.join(batch_path, 'threshold_metrics.csv')
    threshold_metrics.to_csv(threshold_path, index=False)

