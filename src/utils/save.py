'''
Save training artifacts: config, data splits, logs, plots, model.
'''
import os
import torch
import pandas as pd
from pathlib import Path
from utils.timer import Timer
from typing import Dict, List
from .plots import plot_results
from utils.device import main_process_only
from omegaconf import OmegaConf, DictConfig


@main_process_only
def save_before_training(cfg: DictConfig,
                      run_dir: str,
                      df_train: pd.DataFrame,
                      df_val: pd.DataFrame,
                      save_dataframe_dir: str = 'dataframes',
                      save_config_name: str = 'config.yaml') -> None:
    """
    Creates a run/experiment directory and saves config, train_data.csv, and val_data.csv.

    Args:
        cfg: OmegaConf config object containing training settings.
        run_dir: Directory to save all training artifacts.
        df_train: Training dataframe with image paths and labels.
        df_val: Validation dataframe with image paths and labels.
        save_dataframe_dir: Subdirectory name for saving dataframes.
        save_config_name: Filename for the saved config.
    """
    # create experiment directory folders
    os.makedirs(run_dir, exist_ok=True)
    # create dataframe directory folders 
    save_dataframe_dir = os.path.join(run_dir, save_dataframe_dir)
    os.makedirs(save_dataframe_dir, exist_ok=True)

    # write config to the run directory
    with open(os.path.join(run_dir, save_config_name), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Save processed train and test data csv(s)
    df_train.to_csv(os.path.join(save_dataframe_dir, 'train_data.csv'), index=False)
    df_val.to_csv(os.path.join(save_dataframe_dir, 'val_data.csv'), index=False)


@main_process_only
def save_at_epoch_end(run_dir: str, 
                      results: Dict[str, List]) -> None:
    """
    Updates log.csv and the training plots at the end of each epoch.

    Args:
        run_dir: Directory containing all training artifacts.
        results: Dictionary with training metrics (loss, acc, apcer, bpcer) per epoch.
    """
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save training logs
    pd.DataFrame(results).to_csv(os.path.join(run_dir, 'log.csv'), index_label='epoch')

    # Plot training results
    plot_results(results, save_dir=plots_dir)


@main_process_only
def save_after_training(run_dir: str, 
                        ddp_model: torch.nn.Module, 
                        save_name: str, 
                        timer: Timer) -> None:
    """
    Saves final model's state dict and elasped time information after training completes.

    Args:
        run_dir: Directory containing all training artifacts.
        ddp_model: DDP-wrapped model (will be unwrapped via .module).
        save_name: The experiment name (defined in train_config)
        timer: Optional Timer object to record timing info in config.
    """
    # Update config with timing if available
    
    config_path = os.path.join(run_dir, 'config.yaml')
    save_cfg = OmegaConf.load(config_path)
    save_cfg.timing = timer.summary()
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(save_cfg))

    # unwrap the ddp model 
    model = ddp_model.module
    save_name = f"{save_name}.pt"

    # save the model's state dict
    _save_model_state_dict(model=model, target_dir=run_dir, save_name=save_name)


def _save_model_state_dict(model: torch.nn.Module,
                           target_dir: str,
                           save_name: str) -> None:
    """Internal function to save a PyTorch model's state_dict to the experiment directory."""
    model_save_path = Path(target_dir) / save_name
    torch.save(obj=model.state_dict(), f=model_save_path)
