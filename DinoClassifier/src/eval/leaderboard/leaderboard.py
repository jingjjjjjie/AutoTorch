"""
Leaderboard generator for model selection.
Reads eval results and ranks checkpoints by weighted score.

Usage:
    python leaderboard_automl_old.py --config leaderboard_config.json --runs_dir ../../runs
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.model_selection import ModelSelection


def run(config_path: str, runs_dir: str):
    """Run leaderboard generation."""

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_config = config['data']
    leaderboard_config = data_config['leaderboard']

    score_criteria = leaderboard_config['score_criteria']
    model_checkpoint = leaderboard_config['model_checkpoint']
    models = leaderboard_config['models']
    thres_setting = leaderboard_config.get('thres_setting', 'all')

    # Initialize model selection
    model_selection = ModelSelection(score_criteria, 'idFraud')

    run_checkpoint = model_checkpoint is not None

    if run_checkpoint:
        # Single checkpoint mode - evaluate all epochs within one experiment
        print(f"Running leaderboard for checkpoint: {model_checkpoint}")

        best_model, best_model_backup, leaderboard_df = model_selection.run_checkpoint(
            model_checkpoint,
            artifact_dir=runs_dir
        )

        # Save leaderboard
        leaderboard_dir = os.path.join(runs_dir, model_checkpoint, 'eval')
        leaderboard_csvpath = os.path.join(leaderboard_dir, 'leaderboard.csv')

    else:
        # Multiple models mode
        print(f"Running leaderboard for models: {models}")

        model_paths = [
            os.path.join(runs_dir, model, 'eval')
            for model in models
        ]

        model_h5_paths = [
            os.path.join(runs_dir, model)
            for model in models
        ]

        best_model, best_model_backup, leaderboard_df = model_selection.run(
            model_paths,
            model_h5_paths,
            thres_setting=thres_setting
        )

        leaderboard_csvpath = os.path.join(runs_dir, 'leaderboard.csv')

    # Save results
    leaderboard_df.to_csv(leaderboard_csvpath, index=False)
    print(f"\nLeaderboard saved to: {leaderboard_csvpath}")

    # Print summary
    print("\n" + "=" * 70)
    print("LEADERBOARD (sorted by score, higher is better)")
    print("=" * 70)
    print(leaderboard_df.to_string(index=False))
    print("=" * 70)

    print(f"\nBest model: {best_model}")
    if best_model_backup:
        print(f"Best model (with pass criteria): {best_model_backup}")

    return best_model, best_model_backup, leaderboard_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate leaderboard from eval results")
    parser.add_argument("--config", type=str, default="leaderboard_config.json",
                        help="Path to leaderboard config JSON")
    parser.add_argument("--runs_dir", type=str, default="../../runs",
                        help="Path to runs directory")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
    runs_dir = os.path.join(script_dir, args.runs_dir) if not os.path.isabs(args.runs_dir) else args.runs_dir

    run(config_path, runs_dir)
