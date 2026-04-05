import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
    test_dataset = datasets.ImageFolder("dataset/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader