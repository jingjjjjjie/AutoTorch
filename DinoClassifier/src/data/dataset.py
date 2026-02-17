'''
Custom PyTorch dataset for CSV-based image classification.
'''
from PIL import Image
import torch
from torch.utils.data import Dataset


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
        '''Get percentage and counts of the labels.'''
        counts = self.df['label'].value_counts().sort_index()

        label_map = {1: "fraud", 0: "genuine"}
        counts.index = counts.index.map(lambda x: label_map.get(x, str(x)))

        if not percentages:
            return counts

        total = counts.sum()
        return (counts / total).rename("percentage")
