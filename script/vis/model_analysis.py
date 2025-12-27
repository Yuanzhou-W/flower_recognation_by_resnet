import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# ================= 配置 =================
BEST_MODEL_PATH = "../../flower_model_best.pth"
OUTPUT_DIR = "../../analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 加载模型 =================
def load_model(weights_path=None):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if weights_path is not None:
        model.fc = nn.Linear(model.fc.in_features, 5)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ================= 1. 权重分布对比 =================
def plot_weight_distribution():
    print("绘制权重分布对比图...")

    model_pre = load_model(weights_path=None)
    model_post = load_model(weights_path=BEST_MODEL_PATH)

    weights_pre = []
    weights_post = []

    for (n1, p1), (n2, p2) in zip(
        model_pre.named_parameters(),
        model_post.named_parameters()
    ):
        if "weight" in n1 and p1.dim() > 1:
            weights_pre.append(p1.detach().cpu().numpy().ravel())
            weights_post.append(p2.detach().cpu().numpy().ravel())

    weights_pre = np.concatenate(weights_pre)
    weights_post = np.concatenate(weights_post)

    plt.figure(figsize=(8, 5))
    plt.hist(weights_pre, bins=100, alpha=0.5, label="Pretrained")
    plt.hist(weights_post, bins=100, alpha=0.5, label="Fine-tuned")
    plt.legend()
    plt.title("Weight Distribution Comparison")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "weight_distribution.png"))
    plt.close()

# ================= 2. 第一层卷积核可视化 =================
def visualize_first_conv():
    print("可视化第一层卷积核...")

    model = load_model(weights_path=BEST_MODEL_PATH)
    conv1 = model.conv1.weight.data.cpu()

    # 归一化
    conv1 = (conv1 - conv1.min()) / (conv1.max() - conv1.min())

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= conv1.shape[0]:
            break
        kernel = conv1[i]
        kernel = kernel.permute(1, 2, 0)
        ax.imshow(kernel)
        ax.axis("off")

    plt.suptitle("First Conv Layer Filters (Fine-tuned)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "conv1_filters.png"))
    plt.close()

# ================= 主程序 =================
if __name__ == "__main__":
    plot_weight_distribution()
    visualize_first_conv()
    print(f"分析完成，结果保存在 {OUTPUT_DIR}/")
