"""
Generate leaderboard config by scanning checkpoint/eval folders.
Simplified version - just scans folders and assigns default weights.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# Default weights by dataset type (detected from folder name keywords)
DEFAULT_WEIGHTS = {
    'production': {'apcer': 1.0, 'bpcer': 1.0},
    'datacollection': {'apcer': 1.0, 'bpcer': 1.0},
    'issue': {'apcer': 1.0, 'bpcer': 0.0},  # issue sets often only care about apcer
    'test_plan': {'apcer': 1.0, 'bpcer': 1.0},
    'default': {'apcer': 1.0, 'bpcer': 1.0},
}


def detect_dataset_type(folder_name: str) -> str:
    """Detect dataset type from folder name."""
    name_lower = folder_name.lower()
    for dtype in ['production', 'datacollection', 'issue', 'test_plan']:
        if dtype in name_lower:
            return dtype
    return 'default'


def scan_datasets(eval_dir: str) -> List[str]:
    """Scan evaluation directory for dataset folders."""
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        raise FileNotFoundError(f"Directory not found: {eval_dir}")

    # Find dataset folders (skip epoch_* folders, look inside them)
    datasets = set()
    for item in eval_path.iterdir():
        if item.is_dir():
            # If it's an epoch folder, look inside for dataset folders
            if item.name.startswith('epoch_'):
                for dataset_dir in item.iterdir():
                    if dataset_dir.is_dir():
                        datasets.add(dataset_dir.name)
            else:
                datasets.add(item.name)

    return sorted(datasets)


def generate_config(
    experiment_name: str,
    eval_dir: str,
    custom_weights: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: Optional[str] = None
) -> Dict:
    """
    Generate leaderboard config from evaluation directory.

    Args:
        experiment_name: Name of the experiment/model checkpoint
        eval_dir: Path to evaluation directory (e.g., runs/exp1/eval)
        custom_weights: Override weights for specific datasets
                       {'dataset_name': {'apcer': 1.0, 'bpcer': 0.5}}
        output_path: Optional path to save JSON config

    Returns:
        Config dictionary
    """
    datasets = scan_datasets(eval_dir)
    print(f"Found {len(datasets)} datasets in {eval_dir}")

    # Build weight entries
    weight_entries = []
    for dataset in datasets:
        # Use custom weights if provided, else detect from name
        if custom_weights and dataset in custom_weights:
            weights = custom_weights[dataset]
        else:
            dtype = detect_dataset_type(dataset)
            weights = DEFAULT_WEIGHTS[dtype]

        # Add apcer entry
        if weights.get('apcer', 0) != 0:
            weight_entries.append({
                'data_source': dataset,
                'metric': 'apcer',
                'value': weights['apcer'],
            })

        # Add bpcer entry
        if weights.get('bpcer', 0) != 0:
            weight_entries.append({
                'data_source': dataset,
                'metric': 'bpcer',
                'value': weights['bpcer'],
            })

    config = {
        'model_checkpoint': experiment_name,
        'score_criteria': {
            'weight': weight_entries
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {output_path}")

    return config


def print_config_summary(config: Dict):
    """Print summary of config."""
    weights = config['score_criteria']['weight']
    datasets = set(w['data_source'] for w in weights)

    print(f"\nConfig Summary:")
    print(f"  Model: {config['model_checkpoint']}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Weight entries: {len(weights)}")

    # Group by type
    by_type = {}
    for ds in datasets:
        dtype = detect_dataset_type(ds)
        by_type.setdefault(dtype, []).append(ds)

    print(f"\nBy type:")
    for dtype, ds_list in sorted(by_type.items()):
        print(f"  {dtype}: {len(ds_list)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate leaderboard config')
    parser.add_argument('eval_dir', help='Path to evaluation directory')
    parser.add_argument('--name', '-n', required=True, help='Experiment/model name')
    parser.add_argument('--output', '-o', default='leaderboard_config.json', help='Output JSON path')
    args = parser.parse_args()

    config = generate_config(
        experiment_name=args.name,
        eval_dir=args.eval_dir,
        output_path=args.output
    )
    print_config_summary(config)
