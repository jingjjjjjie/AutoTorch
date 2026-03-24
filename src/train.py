"""
DDP Training Script
Run with: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
test with: CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train.py
docker compose run -e CUDA_VISIBLE_DEVICES=2 overlock2 torchrun --nproc_per_node=1 --master_port=29502 train.py
"""
import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.timer import Timer
from utils.config import get_config
from data.idfraud import create_dataloaders
from eval.idfraud.evaluate import run_evaluation
from eval.idfraud.leaderboard import create_leaderboard
from training.trainers import idfraud_trainer
from training.callbacks import build_checkpoint, build_early_stopping
from training import build_loss_fn, build_optimizer, build_lr_scheduler
from utils import save_before_training, save_after_training, save_at_epoch_end
from utils.device import setup_ddp, wrap_model_ddp, cleanup_ddp
from models import build_model

def main():
    timer = Timer()

    # Load config and extract values
    cfg = get_config(name='train_config.yaml')

    # Setup DDP
    local_rank = setup_ddp()
    timer.record("ddp_setup")

    # Process data, get datasets and dataloaders
    train_loader, valid_loader, train_sampler, df_train, df_val = create_dataloaders(
        image_type=cfg.data.image_type,
        train_batches=cfg.data.train_batches,
        train_val_split=cfg.data.train_val_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        persistent_workers=cfg.dataloader.persistent_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=cfg.dataloader.drop_last,
        image_size=cfg.transform.image_size,
        normalize_mean=tuple(cfg.transform.normalize_mean),
        normalize_std=tuple(cfg.transform.normalize_std),
        sample_fraction=cfg.data.get('sample_fraction', 1.0))
    timer.record("data_processing")
    
    # Build model and wrap with DDP
    model = build_model(
        model_name=cfg.model.backbone_name,
        device=local_rank,
        task=cfg.model.get('task', 'classification'),
        head_type=cfg.model.head_type,
        freeze_backbone=cfg.model.freeze_backbone)
    ddp_model = wrap_model_ddp(model, local_rank)

    # Build loss function, optimizer, and learning rate scheduler
    loss_fn = build_loss_fn(cfg.training.loss_fn)
    optimizer = build_optimizer(name=cfg.training.optimizer, model=ddp_model, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Build early stopping and checkpoint
    early_stopping = build_early_stopping(cfg.get('early_stopping', {}))
    checkpoint = build_checkpoint(cfg.get('checkpoint', {}), run_dir=cfg.run_dir)
    timer.record("model_setup")

    # Save config and data splits
    save_before_training(cfg, cfg.run_dir, df_train, df_val)

    # Train using idfraud_trainer
    results = idfraud_trainer.train(
        model=ddp_model,
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.training.epochs,
        device=local_rank,
        scheduler=scheduler,
        sampler=train_sampler,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        output_type=cfg.model.output_type,
        on_epoch_end=lambda history: save_at_epoch_end(cfg.run_dir, history),
    )
    timer.record("training")

    # Run evaluation --> generates a dataframe containing predictions from all epochs
    run_evaluation(cfg)
    create_leaderboard(cfg)
    timer.record("evaluation")

    # Save results-log.csv, update configs, plots, and final model
    save_after_training(cfg.run_dir, ddp_model=ddp_model, save_name=cfg.experiment.save_name, timer=timer)

    cleanup_ddp()

if __name__ == "__main__":
    main()
  
    