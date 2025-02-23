import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import Counter

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

# Count the number of NORMAL vs PNEUMONIA images in training dataset
train_class_counts = Counter([label for _, label in train_dataset])
val_class_counts = Counter([label for _, label in val_dataset])

# Get class label mappings
class_labels = train_dataset.class_to_idx  # {'NORMAL': 0, 'PNEUMONIA': 1}
reverse_class_labels = {v: k for k, v in class_labels.items()}  # {0: 'NORMAL', 1: 'PNEUMONIA'}

# Print dataset details
print(f"Total Training Images: {len(train_dataset)}")
for label, count in train_class_counts.items():
    print(f"  {reverse_class_labels[label]}: {count} images")

print(f"\nTotal Validation Images: {len(val_dataset)}")
for label, count in val_class_counts.items():
    print(f"  {reverse_class_labels[label]}: {count} images")
