"""
DDP Training Script
Run with: torchrun --nproc_per_node=NUM_GPUS train.py
"""
import torch
from engine import train
from postprocess import save_run
from utils.config import get_config
from utils.timer import Timer
from utils.device import setup_ddp, cleanup_ddp, is_main_process, wrap_model_ddp
from components import build_loss_fn, build_optimizer, build_scheduler, build_early_stopping, build_checkpoint
from data_setup import create_dataloaders
from model import CustomClassifierModel


def main():
    timer = Timer()

    # Setup DDP (prints GPU info automatically)
    local_rank = setup_ddp()
    timer.record("ddp_setup")

    # Load config
    cfg = get_config()

    # Process data, Get datasets and dataloaders
    train_loader, valid_loader, train_sampler, df_train, df_val = create_dataloaders(cfg)
    timer.record("data_processing")

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

    # Build early stopping and checkpoint
    early_stopping = build_early_stopping(cfg)
    checkpoint = build_checkpoint(cfg)
    timer.record("model_setup")

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
        sampler=train_sampler,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
    )
    timer.record("training")

    # Save (guard is inside save_run)
    save_run(cfg, results, model, df_train, df_val, timer)

    # Print timing summary (main process only)
    if is_main_process():
        print("=" * 40)
        print(timer.formatted_summary())
        print("=" * 40)

    cleanup_ddp()


if __name__ == "__main__":
    main()
