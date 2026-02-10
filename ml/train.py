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


# calculate weights for each class, based on frequency of appearance
def compute_class_weights(data):
    counts = data.df["label"].value_counts().sort_index().values
    weights = 1 / counts
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item() * inputs.size(0)
        predicted = outputs.argmax(1)
        correct = correct + (predicted == targets).sum().item()
        total = total + inputs.size(0)

    return train_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss = val_loss + loss.item() * inputs.size(0)
        predicted = outputs.argmax(1)
        correct = correct + (predicted == targets).sum().item()
        total = total + inputs.size(0)

    return val_loss / total, correct / total


def train(csv_path, img_dirs, epochs, batch_size, learning_rate=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_loader, val_loader, class_names = build_loaders(csv_path, img_dirs, batch_size=batch_size)

    base_data = ImageDataset(csv_path, img_dirs, transform=None)
    weights = compute_class_weights(base_data).to(device)

    model = build_model(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    cp_path = Path("checkpoints")
    cp_path.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cp_path / f"best_model.pth")

        print(f"Best accuracy: ", best_acc)


if __name__ == "__main__":
    csv_path = Path("../data/raw/HAM10000_metadata.csv")
    img_dirs = [
        Path("../data/raw/HAM10000_images_part_1"),
        Path("../data/raw/HAM10000_images_part_2"),
    ]

train(csv_path, img_dirs, epochs=100, batch_size=32)