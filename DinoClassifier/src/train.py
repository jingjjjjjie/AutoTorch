"""
DDP Training Script
Run with: torchrun --nproc_per_node=NUM_GPUS train.py
"""
import torch
import time
from engine import train
from postprocess import save_run
from utils.config import get_config
from utils.device import setup_ddp, cleanup_ddp, is_main_process, wrap_model_ddp
from model import CustomClassifierModel
from components import build_loss_fn, build_optimizer, build_scheduler
from data_setup import create_dataloaders


def main():
    # Start Timer
    start_time = time.time()

    # Setup DDP (prints GPU info automatically)
    local_rank = setup_ddp()

    # Load config
    cfg = get_config()

    # Get datasets and dataloaders
    train_loader, valid_loader, train_sampler, df_train, df_val = create_dataloaders(cfg)

    # Load the architecture from dinov3's original repository
    dinov3_vits16 = torch.hub.load(cfg['model']['repo_dir'], 'dinov3_vits16', 
                                   source='local', 
                                   weights=cfg['model']['checkpoint_path'])

    # Build model and move to GPU
    model = CustomClassifierModel(backbone=dinov3_vits16, 
                                  output_dim=cfg['model']['hidden_units'], 
                                  freeze=cfg['model']['freeze_backbone']).to(local_rank)
    model = wrap_model_ddp(model, local_rank) # Wrap with DDP

    # Build loss function, optimizer, and scheduler
    loss_fn = build_loss_fn(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Start training
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=valid_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=cfg['training']['epochs'],
        device=local_rank,
        scheduler=scheduler,
        sampler=train_sampler  # Pass sampler for set_epoch()
    )

    # Save (only from main process)
    if is_main_process():
        save_run(cfg, results, model, df_train, df_val, time.time() - start_time)

    cleanup_ddp()


if __name__ == "__main__":
    main()
