import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import Block

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--train_data_dir", type=str, default="/data/train")
parser.add_argument("--eval_data_dir", type=str, default="/data/eval")
args = parser.parse_args()

# Define preprocessing for the images
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# Load the custom dataset
train_dataset = CustomDataset(root_dir="/data/train", transform=preprocess)
test_dataset = CustomDataset(root_dir="/data/test", transform=preprocess)
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = Block()
print(f"Model Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Train the model
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in progress_bar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"Loss": running_loss / len(progress_bar)})
    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")

# Test the model
model.eval()
test_accuracy = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_accuracy += (predicted == labels).sum().item()
test_accuracy /= len(test_dataset)
print(f"Test Accuracy: {test_accuracy}")
