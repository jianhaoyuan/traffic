from ultralytics import YOLO
import torch


def main():
    # 加载模型
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data='C:/Users/26800/Desktop/code/dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        workers=0,  # Windows必须设为0
        device=0,
        cache=False,
        amp=True,
        patience=20,
        exist_ok=True,
        mosaic=0.0,
    )

    # 验证（关键：workers=0）
    print("Running validation...")
    metrics = model.val(workers=0)
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

    # 导出
    print("Exporting to ONNX...")
    model.export(format='onnx')
    print("Done!")


if __name__ == '__main__':
    main()