"""
Trains a PyTorch image classification model using device-agnostic code.
"""

# TODO - Others
# Display dataset size
# GPU Selection
# DDP

# TODO - Eval and other 
# Cropped, Ori model
# Merged model
# Model Selection

# TODO - Training
# Early Stopping
# Learning rate decay

# TODO - Checkpoint and viz
# Visualize Traning plot
# Checkpoint saving
# Save CSV

import torch
from torchvision import transforms

from engine import train
from utils.model import save_model
from utils.config import get_config
from model_builder import CustomClassifierModel, build_loss_fn, build_optimizer
from data_setup import create_dataloaders

# Load config
cfg = get_config()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transform
t = cfg['transform']
transform = transforms.Compose([
    transforms.Resize((t['image_size'], t['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=t['normalize_mean'], std=t['normalize_std'])
])

# Data
train_dataset, valid_dataset, train_loader, valid_loader, class_names = create_dataloaders(
    train_list=cfg['data']['train_batches'],
    transform=transform,
    batch_size=cfg['training']['batch_size']
)

# Dataset stats
print(f"\n{'='*40}")
print(f"Train: {len(train_dataset)} | Val: {len(valid_dataset)}")
print(f"Train distribution:\n{train_dataset.label_counts()}")
print(f"Val distribution:\n{valid_dataset.label_counts()}")
print(f"{'='*40}\n")

# Load the architecture from dinov3's original repository
dinov3_vits16 = torch.hub.load(
    cfg['model']['repo_dir'], 'dinov3_vits16',
    source='local', weights=cfg['model']['checkpoint_path']
)

# build model (build_model.py)
model = CustomClassifierModel(
    backbone_model=dinov3_vits16,
    backbone_model_output_dim=cfg['model']['hidden_units'],
    freeze_backbone=cfg['model']['freeze_backbone']
).to(device)

# Build loss function and optimizer (build_model.py)
loss_fn = build_loss_fn(cfg['training']['loss_fn'])
optimizer = build_optimizer(cfg['training']['optimizer'], model, cfg['training']['learning_rate'])
# start training
train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=valid_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=cfg['training']['epochs'],
    device=device
)

# Save the trained model
save_model(
    model=model,
    target_dir=cfg['model']['save_dir'],
    model_name=cfg['model']['save_name']
)
