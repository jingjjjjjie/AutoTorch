'''
Custom PyTorch dataset for CSV-based image classification.
'''
import torch
from PIL import Image
from torch.utils.data import Dataset


class IDFraudTorchDataset(Dataset):
    """Dataset for loading images and labels from a DataFrame."""

    def __init__(self, df, transform=None, gt_label='label', img_path='path'):
        """
        Args:
            df: DataFrame with image paths and labels.
            transform: Optional transform to apply to images.
            gt_label: Column name for labels.
            img_path: Column name for image paths.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.gt_label = gt_label
        self.img_path = img_path
        self.classes = sorted(df[gt_label].unique().tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load and return (image, label) tuple."""
        row = self.df.iloc[idx]
        img = Image.open(row[self.img_path]).convert('RGB')
        label = row[self.gt_label]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def label_counts(self, percentages=False):
        """Get label distribution as counts or percentages."""
        counts = self.df[self.gt_label].value_counts().sort_index()

        # map numeric labels to readable names
        label_map = {1: "fraud", 0: "genuine"}
        counts.index = counts.index.map(lambda x: label_map.get(x, str(x)))

        if not percentages:
            return counts

        total = counts.sum()
        return (counts / total).rename("percentage")
