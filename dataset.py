import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MedicineDataset(Dataset):
    """
    Custom Dataset Class for Fake vs. Real Medicine Classification.
    Expected Directory Structure:
        root_dir/
            ├── Real/
            └── Fake/
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (containing 'Real' and 'Fake' subfolders).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Fake', 'Real']  # 0: Fake, 1: Real
        self.image_paths = []
        self.labels = []

        # Load all image paths and labels
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                # Check for valid image extensions
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image or handle error appropriately if image is corrupt
            image = Image.new('RGB', (224, 224))

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Configuration for ResNet50 ---
# ResNet50 requires 224x224 input size and specific normalization
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),       # Resize to standard ResNet input
    transforms.RandomHorizontalFlip(),   # Augmentation: Flip horizontally
    transforms.RandomRotation(10),       # Augmentation: Rotate slightly (simulating hand-held photos)
    transforms.ToTensor(),               # Convert to Tensor
    transforms.Normalize(                # Normalize with ImageNet mean/std
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def get_data_loaders(data_dir, batch_size=32, split_ratio=0.8):
    """
    Helper function to create Train and Validation DataLoaders automatically.
    """
    import torch
    from torch.utils.data import random_split

    # 1. Initialize the full dataset
    full_dataset = MedicineDataset(root_dir=data_dir, transform=data_transforms)

    # 2. Split into Train and Validation sets
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.classes

# --- Example Usage ---
if __name__ == "__main__":
    # Replace this path with your actual dataset path from Kaggle
    dataset_path = "./dataset/fake_vs_real" 
    
    # Create loaders
    train_loader, val_loader, class_names = get_data_loaders(dataset_path)

    print(f"Classes found: {class_names}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Fetch one batch to verify
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}") # Should be [32, 3, 224, 224]