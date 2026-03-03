'''
Dataloader creation and dataset summary utilities.
'''
from torch.utils.data import DataLoader, DistributedSampler
from utils.device import is_main_process
from .dataset import IDFraudTorchDataset
from .transforms import get_transform
from .preprocessing import preprocess_csv, split_data, map_path_to_source


def create_dataloaders(cfg):
    """Build train/val dataloaders from config."""
    transform = get_transform(cfg)

    # Read and combine batch CSVs
    main_data = preprocess_csv(
        image_type=cfg['data']['image_type'],
        batch_list=cfg['data']['train_batches'],
        training_mode=True
    )

    # Split into train/val
    data_csv = split_data(main_data, train_val_split=cfg['data']['train_val_split'])

    # Separate train and validation DataFrames
    df_train = data_csv[data_csv['dataset_type'] == 'train'].reset_index(drop=True)
    df_val = data_csv[data_csv['dataset_type'] == 'validation'].reset_index(drop=True)

    # Map paths to source locations
    df_train = map_path_to_source(df_train, training_mode=True)
    df_val = map_path_to_source(df_val, training_mode=True)

    # Use Custom Dataset to create dataset(s)
    train_dataset = IDFraudTorchDataset(df_train, transform=transform)
    valid_dataset = IDFraudTorchDataset(df_val, transform=transform)

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

    # Create loaders
    num_workers = cfg['dataloader']['num_workers']
    prefetch_factor = cfg['dataloader'].get('prefetch_factor', 2)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'],
        sampler=train_sampler, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg['training']['batch_size'],
        sampler=valid_sampler, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    if is_main_process():
        print_dataset_summary(train_dataset, valid_dataset)

    return train_loader, valid_loader, train_sampler, df_train, df_val


def print_dataset_summary(train_dataset, valid_dataset):
    """Print dataset size and label distribution."""
    print(f"\n{'='*40}")
    print("Dataset Summary")
    print(f"{'='*40}")
    print(f"Train: {len(train_dataset)} | Val: {len(valid_dataset)}")
    print(f"{'-'*40}")
    print(f"Train distribution:\n{train_dataset.label_counts()}")
    print(f"{'-'*40}")
    print(f"Val distribution:\n{valid_dataset.label_counts()}")
    print(f"{'='*40}\n")
