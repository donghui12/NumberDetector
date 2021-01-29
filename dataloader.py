import os
import torch
import numpy as np
import pandas as pd
from skimage import io
import torch.utils.data as data
import cv2
from torchvision import datasets


class MnistDataset(data.Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root = root
        self.transform = transform
    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        label = self.landmarks_frame.iloc[idx, 0]
        image = self.landmarks_frame.iloc[idx, 1:]

        image = np.array([image])
        image = image.astype('float').reshape(28, 28)

        label = torch.tensor(label, dtype=torch.int64)
        image = torch.tensor([image, ], dtype=torch.uint8)

        sample = {'label': label, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

"""
train_csv_file = './NumberDetecor/datasets/mnist_train.csv'
root = './NumberDetecor/'
train_dataset = MnistDataset(train_csv_file, root)
print(train_dataset[1]['label'])
"""
