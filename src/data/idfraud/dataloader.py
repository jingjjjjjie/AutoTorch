'''
Dataloader creation and dataset summary utilities.
'''
import pandas as pd
from typing import List, Tuple
from .transforms import get_transform
from .dataset import IDFraudTorchDataset
from utils.device import main_process_only
from torch.utils.data import DataLoader, DistributedSampler
from .preprocessing import preprocess_csv, split_data, map_path_to_source


def create_dataloaders(image_type: str,
                       train_batches: List[str],
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
                       sample_fraction: float = 1.0) -> Tuple[DataLoader, DataLoader, DistributedSampler, pd.DataFrame, pd.DataFrame]:
    """
    Calls to preprocess and transfrom data, then creates train and validation DataLoaders with DDP samplers.

    Args:
        image_type: Type of image to load, supports 'crop' or 'ori' ONLY.
        train_batches: List of batch identifiers to include in training.
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
        sample_fraction: Fraction of data to sample (default 1.0 for all data).

    Returns:
        Tuple of (train_loader, valid_loader, train_sampler, df_train, df_val).
    """

    transform = get_transform(image_size, normalize_mean, normalize_std)

    # preprocess csv: takes in a list of csv(s) to train, check if the csv is valid and present in the system,
    #                 then concat to a single dataframe
    main_data = preprocess_csv(
        image_type=image_type,
        batch_list=train_batches,
        sample_fraction=sample_fraction,
        training_mode=True,
    )

    # splits the data to train and validation splits(adds an additional column in the dataframe) 
    data_csv = split_data(main_data, train_val_split=train_val_split)

    # seperate the splits with the column created from split_data
    df_train = data_csv[data_csv['dataset_type'] == 'train'].reset_index(drop=True)
    df_val = data_csv[data_csv['dataset_type'] == 'validation'].reset_index(drop=True)
    
    # resolve the image paths to actuall paths in our system
    df_train = map_path_to_source(df_train, training_mode=True)
    df_val = map_path_to_source(df_val, training_mode=True)

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
        drop_last=drop_last
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
