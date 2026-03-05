'''
Save training artifacts: config, data splits, logs, plots, model.
'''
import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from utils.device import is_main_process
from .plots import plot_results


def save_pre_training(cfg, df_train, df_val):
    """Save config and data splits before training. Only runs on main process."""
    if not is_main_process():
        return

    run_name = cfg['model']['save_name'].replace('.pth', '')
    run_dir = os.path.join(cfg['model']['save_dir'], run_name)
    dataframe_dir = os.path.join(run_dir, 'dataframes')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(dataframe_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Save data splits
    df_train.to_csv(os.path.join(dataframe_dir, 'train_data.csv'), index=False)
    df_val.to_csv(os.path.join(dataframe_dir, 'val_data.csv'), index=False)

    print(f"[INFO] Saved config and data splits to: {run_dir}")


def save_epoch(cfg, results):
    """Save logs and plots at the end of each epoch. Only runs on main process."""
    if not is_main_process():
        return

    run_name = cfg['model']['save_name'].replace('.pth', '')
    run_dir = os.path.join(cfg['model']['save_dir'], run_name)
    plots_dir = os.path.join(run_dir, 'plots')

    os.makedirs(plots_dir, exist_ok=True)

    # Save training logs
    pd.DataFrame(results).to_csv(os.path.join(run_dir, 'log.csv'), index_label='epoch')

    # Plot training results
    plot_results(results, save_dir=plots_dir)


def save_post_training(cfg, results, model, timer=None):
    """Save training results after training. Only runs on main process."""
    if not is_main_process():
        return

    run_name = cfg['model']['save_name'].replace('.pth', '')
    run_dir = os.path.join(cfg['model']['save_dir'], run_name)
    plots_dir = os.path.join(run_dir, 'plots')

    os.makedirs(plots_dir, exist_ok=True)

    # Update config with timing if available
    if timer:
        config_path = os.path.join(run_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            save_cfg = yaml.safe_load(f)
        save_cfg['timing'] = timer.summary()
        with open(config_path, 'w') as f:
            yaml.dump(save_cfg, f)

    # Save training logs
    pd.DataFrame(results).to_csv(os.path.join(run_dir, 'log.csv'), index_label='epoch')

    # Plot training results
    plot_results(results, save_dir=plots_dir)

    # Save model (unwrap DDP with .module)
    save_model(model=model.module, target_dir=run_dir, model_name=cfg['model']['save_name'])


def save_model(model, target_dir, model_name):
    """Save a PyTorch model's state_dict to target_dir/model_name."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "model_name should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
