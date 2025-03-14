import torch
import polars as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB
        label = torch.tensor([float(self.labels[idx])], dtype=torch.long)  # Binary labels

        if self.transform:
            image = self.transform(image)
            image = image.unsqueeze(0)  # Add batch dimension

        return image, label

def load_data(data: pl.DataFrame) -> DataLoader:
    image_paths = data['path'].to_list()
    labels = data['label'].to_list()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader