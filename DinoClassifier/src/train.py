"""
Trains a PyTorch image classification model using device-agnostic code.
"""
# TOTRY
# Freezing Early Layers
# Differential learning rate (lower lr for pt layers and higher one for new head)

# TODO - Others
# GPU Selection
# DDP

# TODO - Eval and other 
# Cropped, Ori model
# Merged model
# Model Selection

# TODO - Training
# Early Stopping, criteria: 3-5 epochs

# TODO - Checkpoint and viz
# Checkpoint saving

import torch
from torchvision import transforms
import os
import yaml
import pandas as pd
from engine import train, plot_results
from utils.model import save_model
from utils.config import get_config
from model_builder import CustomClassifierModel, build_loss_fn, build_optimizer, build_scheduler
from data_setup import create_dataloaders, print_dataset_summary

# Load config
cfg = get_config()

# Device
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# Transform function for data processing
t = cfg['transform']
transform = transforms.Compose([
    transforms.Resize((t['image_size'], t['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=t['normalize_mean'], std=t['normalize_std'])
])

# Data processing
train_dataset, valid_dataset, train_loader, \
    valid_loader, class_names, df_train, df_val = create_dataloaders(
    train_list=cfg['data']['train_batches'],
    transform=transform,
    batch_size=cfg['training']['batch_size']
)

# Print dataset_summary
print_dataset_summary(train_dataset, valid_dataset)

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
optimizer = build_optimizer(
    cfg['training']['optimizer'],
    model,
    cfg['training']['learning_rate'],
    cfg['training']['weight_decay']
)
scheduler = build_scheduler(
    optimizer,
    warmup_epochs=cfg['scheduler']['warmup_epochs'],
    decay_type=cfg['scheduler']['decay_type'],
    total_epochs=cfg['training']['epochs'],
    step_size=cfg['scheduler']['step_size'],
    gamma=cfg['scheduler']['gamma'],
    eta_min=cfg['scheduler']['eta_min']
)

# start training
results = train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=valid_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=cfg['training']['epochs'],
    device=device,
    scheduler=scheduler
)

# Summary and plots saving 
#--------------------------------------------------------------------------------------------------------------
# create the run directory
run_name = cfg['model']['save_name'].replace('.pth', '')
run_dir = os.path.join(cfg['model']['save_dir'], run_name)
plots_dir = os.path.join(run_dir, 'plots')
dataframe_dir = os.path.join(run_dir, 'dataframes')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(dataframe_dir,exist_ok=True)

# save config
with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
    yaml.dump(cfg, f)

# save data split
df_train.to_csv(os.path.join(dataframe_dir, 'train_data.csv'), index=False)
df_val.to_csv(os.path.join(dataframe_dir, 'val_data.csv'), index=False)

# save the training logs
pd.DataFrame(results).to_csv(os.path.join(run_dir, 'log.csv'), index_label='epoch')

# Plot training results
plot_results(results, save_dir=plots_dir)

# Save the trained model
save_model(model=model, target_dir=run_dir, model_name=cfg['model']['save_name'])
