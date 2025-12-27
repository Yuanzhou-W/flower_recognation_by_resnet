import os
import shutil
import random

# ================= 配置 =================
SRC_ROOT = "../data/flower_data"
SRC_DIRS = ["train", "val"]

DST_ROOT = "../data/flower_data_split"
DST_DIRS = ["train", "val", "test"]

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

# ================= 工具函数 =================
def is_image(fname):
    return fname.lower().endswith((".jpg", ".jpeg", ".png"))

def collect_images():
    images = {}
    for src in SRC_DIRS:
        src_path = os.path.join(SRC_ROOT, src)
        for cls in os.listdir(src_path):
            if cls.startswith("."):
                continue
            cls_path = os.path.join(src_path, cls)
            images.setdefault(cls, [])
            for f in os.listdir(cls_path):
                if is_image(f):
                    images[cls].append(os.path.join(cls_path, f))
    return images

def make_dirs(class_names):
    for split in DST_DIRS:
        for cls in class_names:
            os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

def split_and_copy(images):
    for cls, img_list in images.items():
        random.shuffle(img_list)
        total = len(img_list)

        n_train = int(total * SPLIT_RATIO["train"])
        n_val = int(total * SPLIT_RATIO["val"])

        splits = {
            "train": img_list[:n_train],
            "val": img_list[n_train:n_train + n_val],
            "test": img_list[n_train + n_val:]
        }

        for split, imgs in splits.items():
            for src_path in imgs:
                dst_path = os.path.join(
                    DST_ROOT,
                    split,
                    cls,
                    os.path.basename(src_path)
                )
                shutil.copy2(src_path, dst_path)

# ================= 主程序 =================
if __name__ == "__main__":
    print("开始重新划分数据集（安全模式）...")

    images = collect_images()
    make_dirs(images.keys())
    split_and_copy(images)

    print(f"完成！新数据集已生成于：{DST_ROOT}")
