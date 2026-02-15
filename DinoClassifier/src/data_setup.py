import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from utils.data import read_data, fix_path
from utils.config import get_config
from utils.device import is_main_process
import matplotlib.pyplot as plt

cfg = get_config()

def get_transform():
    """Create transform from config."""
    t = cfg['transform']
    return transforms.Compose([
        transforms.Resize((t['image_size'], t['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=t['normalize_mean'], std=t['normalize_std'])
    ])

def create_dataloaders():
    transform = get_transform()

    # takes in the complete list of csv datasets and return a concatenated csv
    data_csv = read_data('ori', cfg['data']['train_batches'], data_type='train', train_val_split=cfg['data']['train_val_split'], csv_image_column=None)

    # seperate train and validation df(s)
    df_train = data_csv[data_csv['dataset_type'] == 'train'].reset_index(drop=True)
    df_val = data_csv[data_csv['dataset_type'] == 'validation'].reset_index(drop=True)
    
    # map the paths in the csv to their respective locations in the server
    df_train = fix_path(df_train,'train')
    df_val = fix_path(df_val,'validation')

    # Use Custom Dataset to create dataset(s)
    train_dataset = CSVTorchDataset(df_train, transform=transform)
    valid_dataset = CSVTorchDataset(df_val, transform=transform)

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    
    # Get class names
    class_names = train_dataset.classes

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