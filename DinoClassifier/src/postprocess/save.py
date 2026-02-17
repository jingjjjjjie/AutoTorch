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


def save_run(cfg, results, model, df_train, df_val, timer=None):
    """Save all training artifacts. Only runs on main process."""
    if not is_main_process():
        return

    run_name = cfg['model']['save_name'].replace('.pth', '')
    run_dir = os.path.join(cfg['model']['save_dir'], run_name)
    plots_dir = os.path.join(run_dir, 'plots')
    dataframe_dir = os.path.join(run_dir, 'dataframes')

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(dataframe_dir, exist_ok=True)

    # Save config (with timing if available)
    save_cfg = {**cfg}
    if timer:
        save_cfg['timing'] = timer.summary()
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(save_cfg, f)

    # Save data splits
    df_train.to_csv(os.path.join(dataframe_dir, 'train_data.csv'), index=False)
    df_val.to_csv(os.path.join(dataframe_dir, 'val_data.csv'), index=False)

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
