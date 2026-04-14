from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import pandas as pd
import os

# ===================== 1. 初始化配置 =====================
# 加载模型
model = YOLO('runs/detect/train/weights/best.pt')
# 轨迹存储：vehicle_id → 轨迹点队列（保留最近N帧）
trajectories = defaultdict(lambda: deque(maxlen=30))  # 保留30帧历史
# 视频路径配置
VIDEO_INPUT_PATH = 'input/0-DJI_20230427120240_0007_V.MP4'
VIDEO_OUTPUT_PATH = 'output_track2.mp4'
CSV_OUTPUT_PATH = 'output/trajectories_data2.csv'  # 轨迹数据保存路径
os.makedirs('output', exist_ok=True)  # 确保输出目录存在

# ========== 视频预处理优化参数 ==========
TARGET_WIDTH = 1280  # 目标最大宽度
FRAME_SKIP = 2  # 跳帧处理（每N帧处理1帧）
# ========================================

# 模型类别映射
print(f"模型类别: {model.names}")


# ===================== 2. 轨迹转DataFrame函数 =====================
def convert_trajectories_to_df(trajectories):
    """
    将轨迹字典转换为结构化DataFrame，并做基础数据清洗
    """
    track_data = []
    for track_id, track_deque in trajectories.items():
        # 遍历deque中的每个轨迹点（确保完整读取）
        for point in list(track_deque):
            track_data.append({
                'track_id': track_id,
                'frame': point['frame'],
                'time': round(point['time'], 2),  # 时间保留2位小数
                'x': point['pos'][0],
                'y': point['pos'][1],
                'bbox_x1': point['bbox'][0],
                'bbox_y1': point['bbox'][1],
                'bbox_x2': point['bbox'][2],
                'bbox_y2': point['bbox'][3],
                'class': point['class'],
                'conf': round(point['conf'], 3)  # 置信度保留3位小数
            })

    # 转换为DataFrame并排序
    df = pd.DataFrame(track_data)
    if not df.empty:
        # 按帧号+轨迹ID排序，重置索引
        df = df.sort_values(by=['frame', 'track_id']).reset_index(drop=True)
        # 基础数据校验：剔除异常坐标（如负数）
        df = df[(df['x'] >= 0) & (df['y'] >= 0)]
    return df


# ===================== 3. 视频轨迹提取主逻辑（优化版） =====================
def extract_trajectories_from_video():
    # 打开视频
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{VIDEO_INPUT_PATH}")

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ========== 视频预处理优化 ==========
    # 1. 降低视频分辨率
    target_width = min(width, TARGET_WIDTH)
    scale = target_width / width
    target_height = int(height * scale)
    
    # 2. 跳帧处理
    output_fps = fps / FRAME_SKIP
    
    print(f"原始分辨率: {width}x{height}, 目标分辨率: {target_width}x{target_height}")
    print(f"原始帧率: {fps}, 输出帧率: {output_fps}")
    print(f"总帧数: {total_frames}, 处理帧数: {total_frames//FRAME_SKIP}")
    # ====================================

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, output_fps, (target_width, target_height))

    frame_idx = 0
    process_count = 0
    print("开始处理视频，提取轨迹...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # 跳帧处理
        if frame_idx % FRAME_SKIP != 0:
            continue
        
        # 降低分辨率
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # YOLO跟踪（核心）
        results = model.track(
            resized_frame,
            persist=True,  # 保持跟踪器状态
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            verbose=False
        )

        annotated_frame = resized_frame.copy()

        # 处理当前帧的跟踪结果
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            # 遍历每个跟踪目标，存储轨迹
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                # 计算底部中心点（车轮位置，更稳定）
                center_x = (x1 + x2) // 2
                center_y = int(y2 * 0.9)

                # 存储轨迹点
                trajectories[track_id].append({
                    'frame': process_count,
                    'time': process_count / output_fps,
                    'pos': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'class': model.names[int(cls)],
                    'conf': float(conf)
                })

                # 绘制检测框和ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 绘制轨迹线（最近10个点）
            for track_id, track in trajectories.items():
                if len(track) > 1:
                    points = [t['pos'] for t in track]
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i - 1], points[i], (0, 0, 255), 2)

        # 写入标注后的视频帧
        out.write(annotated_frame)
        process_count += 1

        # 进度打印
        if process_count % 50 == 0:
            print(f"已处理 {process_count}/{total_frames//FRAME_SKIP} 帧，当前跟踪 {len(trajectories)} 个目标")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 视频处理完成，共跟踪 {len(trajectories)} 个目标")


# ===================== 4. 结果保存与基础分析 =====================
def save_and_analyze_trajectories(df):
    """
    保存轨迹数据到CSV，并执行基础分析
    """
    if df.empty:
        print("⚠️ 轨迹DataFrame为空，跳过保存和分析")
        return

    # 1. 保存为CSV
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"📊 轨迹数据已保存至：{CSV_OUTPUT_PATH}")

    # 2. 基础数据分析
    print("\n=== 轨迹数据基础分析 ===")
    # 统计每个track_id的轨迹长度（帧数）
    track_length = df.groupby('track_id').size().reset_index(name='轨迹帧数')
    print(f"跟踪目标数量：{len(track_length)}")
    print(f"平均轨迹长度：{track_length['轨迹帧数'].mean():.1f} 帧")
    print(
        f"最长轨迹长度：{track_length['轨迹帧数'].max()} 帧（ID: {track_length.loc[track_length['轨迹帧数'].idxmax(), 'track_id']}）")

    # 统计目标类别分布
    class_dist = df['class'].value_counts()
    print(f"\n目标类别分布：")
    for cls, count in class_dist.items():
        print(f"  {cls}: {count} 个轨迹点")


# ===================== 5. 主执行流程 =====================
if __name__ == "__main__":
    # 步骤1：提取视频轨迹
    extract_trajectories_from_video()

    # 步骤2：转换为DataFrame
    df_trajectories = convert_trajectories_to_df(trajectories)

    # 步骤3：保存并分析
    save_and_analyze_trajectories(df_trajectories)
