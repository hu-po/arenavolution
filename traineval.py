import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

num_epochs = 1
batch_size = 2
learning_rate = 0.001

# Define a simple convolutional network
class ConvNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 56 * 56, num_classes)  # Adjust input size based on your image size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Define preprocessing for the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the ImageNet dataset
train_dataset = ImageNet(root='/data/imagenet', split='train', transform=preprocess)
val_dataset = ImageNet(root='/data/imagenet', split='val', transform=preprocess)
test_dataset = ImageNet(root='/data/imagenet', split='test', transform=preprocess)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader)}')

    # Validate the model
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_accuracy += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_accuracy /= len(val_dataset)
    print(f'Validation Epoch {epoch}, Loss: {val_loss}, Accuracy: {val_accuracy}')

# Test the model
model.eval()
test_accuracy = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_accuracy += (predicted == labels).sum().item()
test_accuracy /= len(test_dataset)
print(f'Test Accuracy: {test_accuracy}')