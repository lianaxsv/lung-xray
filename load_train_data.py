import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Path to the train dataset
TRAIN_PATH = "dataset/chest_xray/train/"

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (for CNNs like ResNet)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.5], [0.5])  # Normalize values
])

# Load only the training dataset
train_dataset = ImageFolder(root=TRAIN_PATH, transform=transform)

# Create DataLoader (to efficiently load data in batches)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print dataset details
print(f"Total Training Images: {len(train_dataset)}")
print(f"Class Labels: {train_dataset.class_to_idx}")  # Should print {'NORMAL': 0, 'PNEUMONIA': 1'}

