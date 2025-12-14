import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "data for vision_stage_2 training")
batch_size = 16
num_epochs = 100
num_classes = 14 #14 screw/nut classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), #50% chance
    transforms.RandomVerticalFlip(), #50% chance
    transforms.RandomRotation(15), #randomly rotates image between -15 and 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #randomly perturbs brightness, contract, saturation, and hue
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #mean and standard deviation per-channel (rgb) of the ImageNet dataset which the backbone is trained on for normalisation around 0
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

#dataset and class to ID
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "Training"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "Validation"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

#training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (preds.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    print(f"train loss: {train_loss:.4f} accuracy: {train_acc:.4f}")

    #validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            val_loss += criterion(preds, labels).item() * images.size(0)
            val_correct += (preds.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    print(f"val loss: {val_loss:.4f} accuracy: {val_acc:.4f}")

#save model
torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_resnet18.pth"))