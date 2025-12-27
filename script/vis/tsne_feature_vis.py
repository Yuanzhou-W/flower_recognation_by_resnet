import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from sklearn.manifold import TSNE

DATA_DIR = "../../data/flower_data_split/test"
MODEL_PATH = "../../flower_model_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds = datasets.ImageFolder(DATA_DIR, transform)
dl = torch.utils.data.DataLoader(ds, batch_size=16)

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(ds.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = nn.Sequential(*list(model.children())[:-1])
model.to(DEVICE)
model.eval()

feats, labels = [], []

with torch.no_grad():
    for x,y in dl:
        x = x.to(DEVICE)
        f = model(x).squeeze()
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())

X = np.concatenate(feats)
y = np.concatenate(labels)

emb = TSNE(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(8,6))
for i, c in enumerate(ds.classes):
    idx = y==i
    plt.scatter(emb[idx,0], emb[idx,1], label=c, alpha=0.7)

plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.title("Feature Space Visualization (After Fine-tuning)")
plt.legend()
plt.grid()
plt.show()
