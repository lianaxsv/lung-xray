import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Define CNN Model (Pretrained ResNet)
class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Updated for latest PyTorch
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust final layer

    def forward(self, x):
        return self.model(x)


# Only execute training if the script is run directly
if __name__ == "__main__":
    from load_train_data import train_loader, val_loader  # Only load when training
    
    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    EPOCHS = 5  # Increase for better accuracy
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}")

    print("Training complete!")

    # Save Model
    torch.save(model.state_dict(), "pneumonia_model.pth")
    print("Model saved as pneumonia_model.pth")
