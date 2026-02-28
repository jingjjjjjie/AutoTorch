"""
DDP Training Script
Run with: torchrun --nproc_per_node=NUM_GPUS train.py
"""
from utils.timer import Timer
from utils.config import get_config
from data.idfraud import create_dataloaders
from models import load_dino_model, build_classifier_model
from postprocess import save_pre_training, save_post_training
from utils.device import setup_ddp, cleanup_ddp, is_main_process, wrap_model_ddp
from training.callbacks import build_checkpoint, build_early_stopping
from training import build_loss_fn, build_optimizer, build_lr_scheduler
from training.trainers import BinaryClassificationTrainer
from training.metrics import BinaryClassificationMetrics

def main():
    timer = Timer()

    # load the config 
    cfg = get_config()

    # Setup DDP 
    local_rank = setup_ddp()
    timer.record("ddp_setup")

    # Process data, Get datasets and dataloaders
    train_loader, valid_loader, train_sampler, df_train, df_val = create_dataloaders(cfg)
    timer.record("data_processing")

    # Build model and wrap with DDP
    backbone = load_dino_model(cfg)
    model = build_classifier_model(cfg, device=local_rank, backbone_model=backbone)
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

    # Setup metrics handler and trainer
    metrics_handler = BinaryClassificationMetrics(threshold=cfg['training']['training_threshold'])
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
    )
    results = trainer.train(train_loader, valid_loader, epochs=cfg['training']['epochs'])
    timer.record("training")

    # Save results, plots, and final model (guard is inside save_post_training)
    save_post_training(cfg, results, model, timer)

    # Print timing summary (main process only)
    if is_main_process():
        timer.print_formatted_summary()

    cleanup_ddp()


if __name__ == "__main__":
    main()
  
    