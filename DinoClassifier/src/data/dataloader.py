'''
Dataloader creation and dataset summary utilities.
'''
from torch.utils.data import DataLoader, DistributedSampler
from utils.device import is_main_process
from .dataset import CSVTorchDataset
from .transforms import get_transform
from .preprocessing import read_data, fix_path


def create_dataloaders(cfg):
    """Build train/val dataloaders from config."""
    transform = get_transform(cfg)

    # takes in the complete list of csv datasets and return a concatenated csv
    data_csv = read_data('ori', cfg['data']['train_batches'], data_type='train',
                         train_val_split=cfg['data']['train_val_split'], csv_image_column=None)

    # separate train and validation df(s)
    df_train = data_csv[data_csv['dataset_type'] == 'train'].reset_index(drop=True)
    df_val = data_csv[data_csv['dataset_type'] == 'validation'].reset_index(drop=True)

    # map the paths in the csv to their respective locations in the server
    df_train = fix_path(df_train, 'train')
    df_val = fix_path(df_val, 'validation')

    # Use Custom Dataset to create dataset(s)
    train_dataset = CSVTorchDataset(df_train, transform=transform)
    valid_dataset = CSVTorchDataset(df_val, transform=transform)

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'],
        sampler=train_sampler, num_workers=cfg['dataloader']['num_workers'], pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg['training']['batch_size'],
        sampler=valid_sampler, num_workers=cfg['dataloader']['num_workers'], pin_memory=True
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
