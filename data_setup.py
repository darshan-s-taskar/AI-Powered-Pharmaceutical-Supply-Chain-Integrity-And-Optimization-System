import torch
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# def download_and_extract(url, zip_path="./processed.zip", extract_dir="./"):
#     gdown.download(url, zip_path, quiet=False)
#     shutil.unpack_archive(zip_path, extract_dir)
#     print(f"Extraction complete to: {extract_dir}")

def create_dataloaders(train_dir, val_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.classes