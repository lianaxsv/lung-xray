import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train import PneumoniaCNN  # Import trained model

# Define dataset path
TEST_PATH = "dataset/chest_xray/test/"  # Path to test images

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load test dataset
test_dataset = ImageFolder(root=TEST_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Load one image at a time

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Class label mapping (from ImageFolder)
class_labels = {v: k for k, v in test_dataset.class_to_idx.items()}  # Reverse class mapping

# Run predictions on test dataset
correct = 0
total = 0

print("\n🔍 **Testing Model on Full Test Dataset:**")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted_class = torch.argmax(outputs, dim=1).item()
        actual_class = labels.item()

        # Print actual vs predicted label
        actual_label = class_labels[actual_class]
        predicted_label = class_labels[predicted_class]
        print(f"Actual: {actual_label}, Predicted: {predicted_label}")

        # Check if the prediction is correct
        if predicted_class == actual_class:
            correct += 1
        total += 1

# Calculate accuracy
accuracy = correct / total * 100
print(f"\n✅ Model Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct predictions)")
