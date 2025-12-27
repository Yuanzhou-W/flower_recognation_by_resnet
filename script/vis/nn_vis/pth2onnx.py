import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torchvision import models

# ================= 配置 =================
PTH_PATH = "../../../flower_model_best.pth"  # 你的 .pth 模型
ONNX_PATH = "flower_model.onnx"  # 输出的 .onnx 文件

NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 构建模型结构 =================
# 必须和训练时的结构【完全一致】
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# ================= 加载权重 =================
state_dict = torch.load(PTH_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ================= 构造假输入 =================
# ResNet18 的标准输入：N x 3 x 224 x 224
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

# ================= 导出 ONNX =================
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,        # 导出参数
    opset_version=11,          # 11 是兼容性最好的选择
    do_constant_folding=True,  # 常量折叠优化
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"ONNX 模型已成功导出至: {ONNX_PATH}")
