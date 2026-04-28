'''
Dataloader creation and dataset summary utilities.
'''
import pandas as pd
from typing import Tuple
from .transforms import build_transform
from .dataset import IDFraudTorchDataset
from .preprocessing import split_data
from utils.device import main_process_only
from torch.utils.data import DataLoader, DistributedSampler


def create_dataloaders(train_csv: str,
                       train_val_split: float,
                       batch_size: int,
                       num_workers: int,
                       prefetch_factor: int,
                       persistent_workers: bool,
                       pin_memory: bool,
                       drop_last: bool,
                       image_size: int,
                       normalize_mean: Tuple[float, ...],
                       normalize_std: Tuple[float, ...],
                       transform_version: str = 'v1') -> Tuple[DataLoader, DataLoader, DistributedSampler, pd.DataFrame, pd.DataFrame]:
    """
    Creates train and validation DataLoaders from a pre-built CSV with DDP samplers.

    The CSV must have:
        - 'path'  : absolute path to the image
        - 'label' : integer label (0 = genuine, 1 = fraud)

    Args:
        train_csv: Path to training CSV (will be split into train/val).
        train_val_split: Fraction of data to use for training.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        prefetch_factor: Number of batches to prefetch per worker.
        persistent_workers: Keep worker processes alive between batches.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop incomplete last batch (required for BatchNorm).
        image_size: Target image size for resizing (square).
        normalize_mean: Mean values for normalization (per channel).
        normalize_std: Std values for normalization (per channel).
        transform_version: Transform pipeline version ('v1', 'v2', 'v3').

    Returns:
        Tuple of (train_loader, valid_loader, train_sampler, df_train, df_val).
    """
    transform = build_transform(image_size, normalize_mean, normalize_std, version=transform_version)

    main_data = pd.read_csv(train_csv)
    data_csv  = split_data(main_data, train_val_split=train_val_split)

    df_train = data_csv[data_csv['dataset_type'] == 'train'].reset_index(drop=True)
    df_val   = data_csv[data_csv['dataset_type'] == 'validation'].reset_index(drop=True)

    # create datasets
    train_dataset = IDFraudTorchDataset(df_train, transform=transform)
    valid_dataset = IDFraudTorchDataset(df_val, transform=transform)

    # create distributed samplers(for ddp) 
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

    # create data loaders with distributed samplers as the sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )

    # print the dataset summary
    _print_dataset_summary(train_dataset, valid_dataset)

    return train_loader, valid_loader, train_sampler, df_train, df_val


@main_process_only
def _print_dataset_summary(train_dataset: IDFraudTorchDataset,
                           valid_dataset: IDFraudTorchDataset) -> None:
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
