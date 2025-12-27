import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import gradio as gr
from PIL import Image
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# ================= 1. 配置参数 =================
DATA_DIR = "../data/flower_data_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_PATH = "../flower_model.pth"
BEST_MODEL_PATH = "../flower_model_best.pth"
LOG_PATH = "training_log.csv"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(CLASSES)

# ================= 2. 数据预处理 =================
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}

# ================= 3. 训练函数（含统计） =================
def train_model():
    print(f"检测到类别：{CLASSES}")

    # ========= 数据 =========
    train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transforms["train"])
    val_dataset = datasets.ImageFolder(VAL_DIR, data_transforms["val"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ========= 模型 =========
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ========= TensorBoard =========
    writer = SummaryWriter("../analysis_results/tensorboard")

    best_val_acc = 0.0

    print("开始训练模型...")
    for epoch in range(EPOCHS):

        # ================== 训练阶段 ==================
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix(loss=train_loss_sum / len(train_loader))

        train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total

        # ================== 验证阶段 ==================
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / len(val_loader)
        val_acc = val_correct / val_total

        # ================== TensorBoard 记录 ==================
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(name, param, epoch)

        # ================== 日志输出 ==================
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # ================== 保存最优模型 ==================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # ========= 收尾 =========
    writer.close()
    torch.save(model.state_dict(), MODEL_PATH)

    print("训练完成，模型已保存")
    return model



# ================= 4. 绘制训练曲线 =================
def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("training_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")
    plt.savefig("training_accuracy.png")
    plt.close()

    print("训练曲线已保存：training_loss.png / training_accuracy.png")

# ================= 5. 推理函数 =================
def predict_flower(img):
    if img is None:
        return "请上传一张图片"

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    weight_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    transform = data_transforms["val"]
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs[0], dim=0)

    return {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

# ================= 6. Gradio GUI =================
def launch_gui():
    interface = gr.Interface(
        fn=predict_flower,
        inputs=gr.Image(type="pil", label="上传花卉图片"),
        outputs=gr.Label(num_top_classes=3, label="识别结果"),
        title="🌸 智能花卉识别系统",
        description="基于 ResNet18 的花卉分类模型（含训练统计分析）"
    )
    interface.launch(share=True)

# ================= 7. 主入口 =================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    launch_gui()
