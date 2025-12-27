import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# ================= 配置 =================
IMAGE_PATH = "../../data/flower_data_split/test/daisy/521762040_f26f2e08dd.jpg"  # ← 换成真实图片
MODEL_PATH = "../../flower_model_best.pth"
OUTPUT_DIR = "../../analysis_results/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型 =================
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ================= Hook（新版写法） =================
feature_maps = {}
gradients = {}

def forward_hook(module, input, output):
    feature_maps["value"] = output

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0]

# 使用 full backward hook（官方推荐）
model.layer4.register_forward_hook(forward_hook)
model.layer4.register_full_backward_hook(backward_hook)

# ================= 图像 =================
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

# ================= 前向 + 反向 =================
output = model(input_tensor)
pred_class = output.argmax(dim=1)

model.zero_grad()
output[0, pred_class].backward()

# ================= Grad-CAM 计算 =================
fmap = feature_maps["value"].detach().cpu().numpy()
grads = gradients["value"].detach().cpu().numpy()

weights = grads.mean(axis=(2, 3))[0]
cam = np.zeros(fmap.shape[2:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * fmap[0, i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, img.size)
cam = cam / (cam.max() + 1e-8)

# ================= 可视化 =================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.colorbar(label="Activation Intensity")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.title(f"Overlay (Predicted: {CLASSES[pred_class]})")
plt.axis("off")

plt.suptitle("Grad-CAM Visualization of Model Attention", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gradcam.png", dpi=300)
plt.show()
