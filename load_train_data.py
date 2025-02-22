import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define dataset paths
TRAIN_PATH = "dataset/chest_xray/train/"
VAL_PATH = "dataset/chest_xray/val/"  # Include validation dataset

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to PyTorch tensors
    transforms.Normalize([0.5], [0.5])  # Normalize values
])

# Load training dataset
train_dataset = ImageFolder(root=TRAIN_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load validation dataset
val_dataset = ImageFolder(root=VAL_PATH, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print dataset details
print(f"Total Training Images: {len(train_dataset)}")
print(f"Total Validation Images: {len(val_dataset)}")
print(f"Class Labels: {train_dataset.class_to_idx}")  # Should print {'NORMAL': 0, 'PNEUMONIA': 1'}
