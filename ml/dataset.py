from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        img_dirs,
        transform=None,
        mode="train"
    ):
        self.df = pd.read_csv(csv_path)

        self.df["age"] = self.df["age"].fillna(self.df["age"].median())

        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.transform = transform
        self.mode = mode

        self.class_names = sorted(self.df["dx"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.df["label"] = self.df["dx"].map(self.class_to_idx)

        self.image_map = self._build_image_map()

    def _build_image_map(self):
        image_map = {}
        for d in self.img_dirs:
            for p in Path(d).glob("*.jpg"):
                image_map[p.stem] = p
        return image_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]

        img_path = self.image_map.get(img_id)
        if img_path is None:
            raise FileNotFoundError(img_id)

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = row["label"]
        return img, label
