import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import yaml
from model import Block

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--child_name", type=str, default="test")
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--train_data_dir", type=str, default="/data/train")
parser.add_argument("--test_data_dir", type=str, default="/data/test")
parser.add_argument("--ckpt_dir", type=str, default="/ckpt")
parser.add_argument("--logs_dir", type=str, default="/logs")
args = parser.parse_args()

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


train_dataset = CustomDataset(root_dir=args.train_data_dir, transform=preprocess)
test_dataset = CustomDataset(root_dir=args.test_data_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

num_classes = len(train_dataset.dataset.classes)
model = Block(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

writer = SummaryWriter(log_dir=args.logs_dir)
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({"Loss": running_loss / len(progress_bar)})
    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")

    writer.add_scalar("Loss/Train", running_loss / len(train_loader), epoch)

    model.eval()
    test_accuracy = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_accuracy += (predicted == labels).sum().item()
    test_accuracy /= len(test_dataset)
    print(f"Test Accuracy: {test_accuracy}")

    writer.add_scalar("Accuracy/Test", test_accuracy, epoch)

    torch.save(model.state_dict(), f"{args.ckpt_dir}/epoch.{epoch}.pth")

hparams = {
    "num_epochs": args.num_epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "model_size": sum(p.numel() for p in model.parameters()),
}
scores = {
    "test_accuracy": test_accuracy,
}
writer.add_hparams(hparams, scores)
writer.close()
results_filepath = os.path.join(args.ckpt_dir, "results.yaml")
if os.path.exists(results_filepath):
    with open(results_filepath, "r") as f:
        results = yaml.safe_load(f) or {}
else:
    results = {}

results[args.child_name] = hparams.update(scores)
with open(results_filepath, "w") as f:
    yaml.dump(results, f, default_flow_style=False)
