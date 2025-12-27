import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATA_DIR = "../../data/flower_data_split/train"
VAL_DIR = "../../data/flower_data_split/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(DATA_DIR, transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)

def train_model(model):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    acc_list = []

    for _ in range(EPOCHS):
        model.train()
        for x,y in train_dl:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)

        acc_list.append(correct/total)

    return acc_list

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Linear(resnet.fc.in_features, 5)

mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 5)

acc_r = train_model(resnet)
acc_m = train_model(mobilenet)

plt.plot(acc_r, label="ResNet18")
plt.plot(acc_m, label="MobileNetV2")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Backbone Comparison")
plt.legend()
plt.grid()
plt.show()
