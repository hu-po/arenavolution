import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import yaml
from model import Block

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="test1")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)
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
        # TODO: Normalize to generated dataset?
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = ImageFolder(root=args.train_data_dir, transform=preprocess)
test_dataset = ImageFolder(root=args.test_data_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
model = Block(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

tblog_filepath = os.path.join(args.logs_dir, args.run_name)
writer = SummaryWriter(tblog_filepath)
print(f"Writing logs to {tblog_filepath}")
hparams = {
    "num_epochs": args.num_epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "model_size": sum(p.numel() for p in model.parameters()),
}
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
        progress_bar.set_postfix({"loss": running_loss / len(progress_bar)})
    print(f"epoch {epoch}, loss: {running_loss / len(train_loader)}")
    writer.add_scalar("loss/train", running_loss / len(train_loader), epoch)
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
    print(f"acc/test: {test_accuracy}")
    writer.add_scalar("acc.test", test_accuracy, epoch)
torch.save(model.state_dict(), f"{args.ckpt_dir}/{args.run_name}.e{epoch}.pth")
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
hparams.update(scores)
results[args.run_name] = hparams
with open(results_filepath, "w") as f:
    yaml.dump(results, f)
