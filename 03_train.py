from ultralytics import YOLO
import torch
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # 清除缓存文件
    cache_files = [
        'dataset/labels/train.cache',
        'dataset/labels/val.cache'
    ]
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info(f"已清除缓存文件: {cache_file}")

    # 加载模型 - 使用稍大的模型提高精度
    model = YOLO('yolov8s.pt')  # 使用small版本，平衡精度和速度

    # 训练参数优化
    results = model.train(
        data='D:/code/dataset/data.yaml',
        epochs=150,  # 增加训练轮数
        imgsz=640,
        batch=8,  # 增加批次大小（如果硬件允许）
        workers=0,  # Windows必须设为0
        device=0,
        cache=True,  # 启用缓存加速训练
        amp=True,  # 启用自动混合精度
        patience=30,  # 增加早停 patience
        exist_ok=True,
        mosaic=0.5,  # 启用马赛克数据增强
        mixup=0.2,  # 启用混合数据增强
        degrees=10.0,  # 旋转增强
        translate=0.1,  # 平移增强
        scale=0.1,  # 缩放增强
        shear=0.0,  # 剪切增强
        flipud=0.0,  # 上下翻转
        fliplr=0.5,  # 左右翻转
        box=7.5,  # 边界框损失权重
        cls=0.5,  # 分类损失权重
        dfl=1.5,  # 分布焦点损失权重
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,  # 预热轮数
        warmup_momentum=0.8,  # 预热动量
        warmup_bias_lr=0.1,  # 预热偏置学习率
        cos_lr=True,  # 使用余弦退火学习率
    )

    # 验证（关键：workers=0）
    logger.info("Running validation...")
    metrics = model.val(workers=0)
    logger.info(f"mAP50: {metrics.box.map50:.3f}")
    logger.info(f"mAP50-95: {metrics.box.map:.3f}")
    logger.info(f"Precision: {metrics.box.mp:.3f}")
    logger.info(f"Recall: {metrics.box.mr:.3f}")

    # 导出多个格式
    logger.info("Exporting models...")
    # 导出ONNX格式
    model.export(format='onnx')
    # 导出TorchScript格式
    model.export(format='torchscript')
    # 导出TensorRT格式（如果支持）
    try:
        model.export(format='engine')
    except Exception as e:
        logger.warning(f"TensorRT export failed: {str(e)}")
    
    logger.info("Model training and export completed!")


if __name__ == '__main__':
    main()