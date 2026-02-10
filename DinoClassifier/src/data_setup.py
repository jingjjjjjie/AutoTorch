import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data import read_data, fix_path
import matplotlib.pyplot as plt

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

    return train_dataset, validation_dataset, train_dataloader, validation_dataloader, class_names, df_train, df_val

def print_dataset_summary(train_dataset, valid_dataset):
    """Print dataset size and label distribution."""
    print(f"\n{'='*40}")
    print("Dataset Summary")
    print(f"\n{'='*40}")
    print(f"Train: {len(train_dataset)} | Val: {len(valid_dataset)}")
    print(f"Train distribution:\n{train_dataset.label_counts()}")
    print(f"Val distribution:\n{valid_dataset.label_counts()}")
    print(f"{'='*40}\n")


def visualize_sample_image_from_dataloader(dataloader):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    true_img = None
    false_img = None

    # Loop until we find both classes
    for images, labels in dataloader:
        for i in range(len(labels)):
            if labels[i] == 1 and true_img is None:
                true_img = images[i]

            if labels[i] == 0 and false_img is None:
                false_img = images[i]

            if true_img is not None and false_img is not None:
                break
        if true_img is not None and false_img is not None:
            break

    def prep(img):
        img = img * std + mean
        img = img.permute(1,2,0).numpy()
        return img.clip(0,1)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(prep(false_img))
    plt.title("Genuine (0)")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(prep(true_img))
    plt.title("Fraud (1)")
    plt.axis('off')

    plt.show()