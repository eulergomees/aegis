from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import models

from ml.dataset import ImageDataset
from ml.transforms import train_transform, val_transform


def build_loaders(csv_path, img_dirs, batch_size=32, num_workers=4):
    base_data = ImageDataset(csv_path, img_dirs, transform=None)
    labels = base_data.df["label"].values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

    train_data = Subset(ImageDataset(csv_path, img_dirs, transform=train_transform), train_idx)
    val_data = Subset(ImageDataset(csv_path, img_dirs, transform=val_transform), val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, base_data.class_names


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

#calculate weights for each class, based on frequency of appearance
def compute_class_weights(data):
    counts = data.df["label"].value_counts().sort_index().values
    weights = 1 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)
