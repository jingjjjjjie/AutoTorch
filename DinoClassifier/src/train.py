"""
DDP Training Script
Run with: torchrun --nproc_per_node=NUM_GPUS train.py
"""
from postprocess import save_pre_training, save_post_training
from utils.config import get_config
from utils.timer import Timer
from utils.device import setup_ddp, cleanup_ddp, is_main_process, wrap_model_ddp
from data import create_dataloaders
from models import build_classifier_model
from training import (
    train,
    build_loss_fn,
    build_optimizer,
    build_lr_scheduler,
    build_early_stopping,
    build_checkpoint,
)

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

    # Build model and wrap with DDP
    model = build_classifier_model(cfg, local_rank)
    model = wrap_model_ddp(model, local_rank)

    # Build loss function, optimizer, and scheduler
    loss_fn = build_loss_fn(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Build early stopping and checkpoint
    early_stopping = build_early_stopping(cfg)
    checkpoint = build_checkpoint(cfg)
    timer.record("model_setup")

    # Save config and data splits before training
    save_pre_training(cfg, df_train, df_val)

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

    # Save results, plots, and model (guard is inside save_post_training)
    save_post_training(cfg, results, model, timer)

    # Print timing summary (main process only)
    if is_main_process():
        print("=" * 40)
        print(timer.formatted_summary())
        print("=" * 40)

    cleanup_ddp()


if __name__ == "__main__":
    main()
