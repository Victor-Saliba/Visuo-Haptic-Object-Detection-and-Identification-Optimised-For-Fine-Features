import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "data for gs training")
batch_size = 16
num_epochs = 100
num_classes = 15 #14 screw/nut classes + 1 'nothing' class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transforms
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #mean and standard deviation per-channel (rgb) of the ImageNet dataset which the backbone is trained on for normalisation around 0
                         std=[0.229, 0.224, 0.225])
])

#dataset and class to ID
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Training"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Validation"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#model
class FlexibleResNet(nn.Module):
    def __init__(self, num_classes=15):  
        super().__init__()
        self.backbone = models.resnet18(pretrained=True) #not frozen
        self.backbone.fc = nn.Identity() #removes original classification layer from backbone
        self.classifier = nn.Linear(512, num_classes) #new classification head

    def forward(self, x):
        feats = self.backbone(x)  #output is [batch size, 512]
        return self.classifier(feats)

model = FlexibleResNet(num_classes=num_classes).to(device)

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
    print(f"train Loss: {train_loss:.4f} accuracy: {train_acc:.4f}")

    #validation
    model.eval()
    val_correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            val_loss += criterion(preds, labels).item() * images.size(0)
            val_correct += (preds.argmax(1) == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    print(f"val Loss: {val_loss:.4f} accuracy: {val_acc:.4f}")

#save model
torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "gs_resnet18_fullres.pth"))