import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ================== 配置 ==================
IMAGE_PATH = "../../../data/flower_data_split/test/daisy/413815348_764ae83088.jpg"  # 换成你的图片
MODEL_PATH = "../../../flower_model_best.pth"
SAVE_PATH = "analysis_results/feature_evolution.gif"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# ================== 模型 ==================
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ================== 图像 ==================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ================== 选取层 ==================
layers = [
    ("Conv1", model.conv1),
    ("Layer1", model.layer1),
    ("Layer2", model.layer2),
    ("Layer3", model.layer3),
    ("Layer4", model.layer4),
]

features = []

x = input_tensor
with torch.no_grad():
    for name, layer in layers:
        x = layer(x)
        fmap = x[0, 0].detach().cpu().numpy()  # 取第 1 个通道
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        features.append((name, fmap))

# ================== Matplotlib Animation ==================
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(features[0][1], cmap="viridis")
title = ax.set_title("")
ax.axis("off")

def update(frame):
    layer_name, fmap = features[frame]
    im.set_data(fmap)
    title.set_text(f"Feature Map Evolution\nLayer: {layer_name}")
    return [im, title]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(features),
    interval=1000,   # 每帧 1 秒
    blit=False
)

# ================== 保存 ==================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
ani.save(SAVE_PATH, writer="pillow", dpi=200)

print(f"动画已保存至：{SAVE_PATH}")
plt.close()
