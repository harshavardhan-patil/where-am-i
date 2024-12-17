import os
import pandas as pd
from torchvision.io import decode_image, read_file
from torch.utils.data import Dataset
import torch
from pathlib import Path

class OSVImageDataset(Dataset):
    def __init__(self, annotations_df, img_dir, transform=None, target_transform=None):
        self.device = torch.device("cuda")
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #todo: idx using image id?
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.jpg')
        image = decode_image(img_path).float() / 255.0
        label = torch.tensor((self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 3]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image.clamp(0, 1)
        return image, label