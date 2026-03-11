"""
DDP Training Script
Run with: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
"""
import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.timer import Timer
from utils.config import get_config
from data.idfraud import create_dataloaders
from models import load_dino_model, build_classifier_model
from training.metrics import BinaryClassificationMetrics
from training.trainers import BinaryClassificationTrainer
from training.callbacks import build_checkpoint, build_early_stopping
from training import build_loss_fn, build_optimizer, build_lr_scheduler
from postprocess import save_pre_training, save_post_training, save_epoch
from eval.idfraud.evaluate import run_evaluation
from utils.device import setup_ddp, cleanup_ddp, is_main_process, wrap_model_ddp

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
        transform_cfg=cfg.transform)
    timer.record("data_processing")
    
    # Build model and wrap with DDP
    backbone, backbone_dim = load_dino_model(cfg.model.backbone_name)
    model = build_classifier_model(
        device=local_rank,
        backbone_model=backbone,
        input_dim=backbone_dim,
        head_type=cfg.model.head_type,
        freeze_backbone=cfg.model.freeze_backbone)
    ddp_model = wrap_model_ddp(model, local_rank) # wrap model with ddp

    # Build loss function, optimizer, and learning rate scheduler
    loss_fn = build_loss_fn(cfg.training.loss_fn)
    optimizer = build_optimizer(name=cfg.training.optimizer, model=ddp_model, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Build early stopping and checkpoint
    early_stopping = build_early_stopping(cfg.get('early_stopping', {}))
    checkpoint = build_checkpoint(cfg.get('checkpoint', {}), run_dir=cfg.run_dir)
    timer.record("model_setup")

    # Save config and data splits 
    save_pre_training(cfg, cfg.run_dir, df_train, df_val)

    # Setup metrics handler and trainer
    metrics_handler = BinaryClassificationMetrics(threshold=cfg.training.training_threshold)
    trainer = BinaryClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics_handler=metrics_handler,
        device=local_rank,
        scheduler=scheduler,
        sampler=train_sampler,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        on_epoch_end=lambda history: save_epoch(cfg.run_dir, history),
    )
    results = trainer.train(train_loader, valid_loader, epochs=cfg.training.epochs)
    timer.record("training")

    # Save results, plots, and final model
    save_post_training(cfg.run_dir, results, model, cfg.experiment.save_name, timer=timer)

    # Run evaluation --> generates a dataframe containing predictions from all epochs
    run_evaluation(cfg)
    timer.record("evaluation")

    cleanup_ddp()

if __name__ == "__main__":
    main()
  
    