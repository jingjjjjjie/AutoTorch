import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import read_data, fix_path

NUM_WORKERS = os.cpu_count() -1 

class CSVTorchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(df['label'].unique().tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        label = row['label']
        
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
    
    def label_counts(self, percentages=False):
        '''Custom helper function to get percentage and counts of the labels'''
        counts = self.df['label'].value_counts().sort_index()

        # Map numeric labels to readable names
        label_map = {1: "fraud", 0: "genuine"}
        counts.index = counts.index.map(lambda x: label_map.get(x, str(x)))

        if not percentages:
            return counts

        total = counts.sum()
        return (counts / total).rename("percentage")

def create_dataloaders(
    train_list,
    transform, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS,
    train_val_split=0.9
):
    training_data = read_data('ori', train_list, data_type='train', train_val_split=train_val_split, csv_image_column=None)

    print("Splitting train/val...")
    df_train = training_data[training_data['dataset_type'] == 'train'].reset_index(drop=True)
    df_val = training_data[training_data['dataset_type'] == 'validation'].reset_index(drop=True)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")

    # map the paths in the csv to their respective locations in the server
    print("Fixing train paths...")
    df_train = fix_path(df_train,'train')
    print("Fixing val paths...")
    df_val = fix_path(df_val,'validation')

    # Use Custom Dataset to create dataset(s)
    train_dataset = CSVTorchDataset(df_train, transform=transform)
    validation_dataset = CSVTorchDataset(df_val, transform=transform)

    # Get class names
    class_names = train_dataset.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, validation_dataset, train_dataloader, validation_dataloader, class_names