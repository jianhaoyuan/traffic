import os
import random
import shutil
import argparse


def split_dataset(source_dir, train_dir, val_dir, val_ratio=0.2):
    """
    将图片随机划分为训练集和验证集

    参数:
        source_dir: 抽帧后的图片目录
        train_dir:  训练集输出目录
        val_dir:    验证集输出目录
        val_ratio:  验证集比例
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有图片
    images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    # 计算划分点
    split_idx = int(len(images) * (1 - val_ratio))

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # 复制文件
    for img in train_images:
        shutil.copy2(os.path.join(source_dir, img), os.path.join(train_dir, img))

    for img in val_images:
        shutil.copy2(os.path.join(source_dir, img), os.path.join(val_dir, img))

    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_images)} 张 → {train_dir}")
    print(f"  验证集: {len(val_images)} 张 → {val_dir}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument("--source", type=str, default="frames/2", help="抽帧后的图片目录")
    parser.add_argument("--train", type=str, default="dataset/images/train", help="训练集输出目录")
    parser.add_argument("--val", type=str, default="dataset/images/val", help="验证集输出目录")
    parser.add_argument("--ratio", type=float, default=0.3, help="验证集比例")
    args = parser.parse_args()
    
    # 运行数据集划分
    split_dataset(
        source_dir=args.source,
        train_dir=args.train,
        val_dir=args.val,
        val_ratio=args.ratio
    )
