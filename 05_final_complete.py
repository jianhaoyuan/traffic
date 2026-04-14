from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")  # 屏蔽所有警告
import matplotlib.pyplot as plt
# 确保matplotlib能正确显示中文
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Arial"]  # 优先使用中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ==============================================
# 🔥 任务1/2/3 核心配置（无人机交叉口专用）
# ==============================================
CONFIG = {
    # 模型/路径配置
    "model_path": 'runs/detect/train/weights/best.pt',
    "video_input": 'input/test_video.mp4',
    "video_output": 'output/final_annotated_video.mp4',
    "traj_csv": 'output/trajectories.csv',
    "conflict_csv": 'output/conflicts.csv',
    "heatmap_path": 'output/conflict_heatmap.png',
    "report_path": 'output/safety_evaluation.txt',

    # 无人机标定（核心：像素→米，车道标准宽3.7m）
    "pixel_per_meter": 30,
    "max_trajectory_len": 60,
    "conf_threshold": 0.25,  # 降低置信度阈值，提高检出率
    "iou_threshold": 0.5,

    "TTC_THRESHOLD": 1.5,  # 进一步调整TTC阈值
    "PET_THRESHOLD": 0.3,  # 进一步调整PET阈值
    "DISTANCE_THRESHOLD": 2.0,  # 进一步调整距离阈值，减少误报
    
    # ========== 视频预处理优化参数 ==========
    "target_width": 1280,  # 目标最大宽度
    "frame_skip": 2,  # 跳帧处理（每N帧处理1帧）
    # ========================================
}

# 机非分类定义（必须与你的模型实际类别一致！）
# 机动车类别
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'vehicle', 'motor']
# 非机动车类别
NON_VEHICLE_CLASSES = ['motorcycle', 'bicycle', 'bike', 'person', 'non_motor']

# 全局变量
trajectories = defaultdict(lambda: deque(maxlen=CONFIG["max_trajectory_len"]))
conflict_records = []
np.random.seed(42)
track_colors = defaultdict(lambda: tuple(map(int, np.random.randint(50, 255, 3))))

# 交通流量统计类
class TrafficFlowAnalyzer:
    def __init__(self):
        self.vehicle_count = 0
        self.non_vehicle_count = 0
        self.vehicle_ids = set()
        self.non_vehicle_ids = set()
        self.flow_by_minute = {}
        self.speed_stats = {}
        
    def update_flow(self, track_id, class_name, current_time):
        """更新交通流量统计"""
        # 按分钟统计
        minute_key = int(current_time // 60)
        
        if minute_key not in self.flow_by_minute:
            self.flow_by_minute[minute_key] = {'vehicle': 0, 'non_vehicle': 0}
        
        # 直接使用类别名称进行判断，因为模型只有motor和non_motor两个类别
        if class_name.lower() == 'motor':
            if track_id not in self.vehicle_ids:
                self.vehicle_ids.add(track_id)
                self.vehicle_count += 1
                self.flow_by_minute[minute_key]['vehicle'] += 1
        elif class_name.lower() == 'non_motor':
            if track_id not in self.non_vehicle_ids:
                self.non_vehicle_ids.add(track_id)
                self.non_vehicle_count += 1
                self.flow_by_minute[minute_key]['non_vehicle'] += 1
    
    def update_speed_stats(self, class_name, speed):
        """更新速度统计"""
        if class_name not in self.speed_stats:
            self.speed_stats[class_name] = []
        self.speed_stats[class_name].append(speed)
    
    def get_total_flow(self):
        """获取总流量"""
        return {
            'vehicle': self.vehicle_count,
            'non_vehicle': self.non_vehicle_count,
            'total': self.vehicle_count + self.non_vehicle_count
        }
    
    def get_flow_by_minute(self):
        """获取按分钟统计的流量"""
        return self.flow_by_minute
    
    def get_speed_statistics(self):
        """获取速度统计信息"""
        stats = {}
        for class_name, speeds in self.speed_stats.items():
            if speeds:
                stats[class_name] = {
                    'average': round(sum(speeds) / len(speeds), 2),
                    'max': round(max(speeds), 2),
                    'min': round(min(speeds), 2)
                }
        return stats
    
    def save_flow_data(self, output_dir='output'):
        """保存流量数据到CSV文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存总流量
        flow_data = []
        for minute, counts in self.flow_by_minute.items():
            flow_data.append({
                'minute': minute,
                'vehicle_count': counts['vehicle'],
                'non_vehicle_count': counts['non_vehicle'],
                'total_count': counts['vehicle'] + counts['non_vehicle']
            })
        
        df_flow = pd.DataFrame(flow_data)
        df_flow.to_csv(os.path.join(output_dir, 'traffic_flow.csv'), index=False, encoding='utf-8')
        
        # 保存速度统计
        speed_data = []
        for class_name, stats in self.get_speed_statistics().items():
            speed_data.append({
                'class': class_name,
                'average_speed': stats['average'],
                'max_speed': stats['max'],
                'min_speed': stats['min']
            })
        
        df_speed = pd.DataFrame(speed_data)
        df_speed.to_csv(os.path.join(output_dir, 'speed_stats.csv'), index=False, encoding='utf-8')
        
        return self.get_total_flow(), self.get_speed_statistics()


# ==============================================
# 工具函数
# ==============================================
def create_output_dir():
    """创建输出文件夹"""
    os.makedirs('output', exist_ok=True)


def pixel2meter(pixel):
    """像素转米（无人机标定）"""
    return round(pixel / CONFIG["pixel_per_meter"], 2)


def calculate_pet(track1, track2):
    """计算PET（侵占时间）"""
    try:
        # 确保轨迹有足够的点
        if len(track1) < 2 or len(track2) < 2:
            return 999
        
        # 简化版PET计算：计算两个目标到达同一区域的时间差
        # 获取两个目标的当前位置和时间
        pos1 = track1[-1]['pos']
        pos2 = track2[-1]['pos']
        time1 = track1[-1]['time']
        time2 = track2[-1]['time']
        
        # 计算距离
        dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        
        # 如果距离小于阈值，计算PET
        if dist < 100:  # 100像素的阈值
            pet = abs(time1 - time2)
            return round(pet, 2)
        else:
            return 999
    except:
        return 999


def calculate_velocity(track):
    """从轨迹计算速度（km/h）"""
    if len(track) < 2:
        return 0
    
    # 取最近两帧计算速度
    pos1 = track[-2]['pos']
    pos2 = track[-1]['pos']
    time1 = track[-2]['time']
    time2 = track[-1]['time']
    time_diff = time2 - time1
    
    if time_diff <= 0:
        return 0
    
    dx = pixel2meter(abs(pos2[0] - pos1[0]))
    dy = pixel2meter(abs(pos2[1] - pos1[1]))
    distance = math.hypot(dx, dy)
    speed_mps = distance / time_diff
    return round(speed_mps * 3.6, 2)


def get_conflict_type(obj1_cls, obj2_cls, angle, track1, track2):
    """判定机非冲突类型"""
    # 直接使用类别名称进行判断，因为模型只有motor和non_motor两个类别
    is_vehicle_1 = obj1_cls.lower() == 'motor'
    is_vehicle_2 = obj2_cls.lower() == 'motor'
    is_non_vehicle_1 = obj1_cls.lower() == 'non_motor'
    is_non_vehicle_2 = obj2_cls.lower() == 'non_motor'

    # 机非冲突判断（一个机动车+一个非机动车）
    if not ((is_vehicle_1 and is_non_vehicle_2) or (is_non_vehicle_1 and is_vehicle_2)):
        return "非机非冲突"

    # 计算速度
    speed1 = calculate_velocity(track1)
    speed2 = calculate_velocity(track2)

    # 计算相对方向和速度差异
    direction_diff = abs(angle - 180)
    speed_diff = abs(speed1 - speed2)

    # 基于角度、速度和方向判定冲突类型
    if 70 < angle < 110:
        if speed1 > 30 or speed2 > 30:
            return "高速直行-横穿冲突"
        else:
            return "直行-横穿冲突"
    elif 160 < angle < 200:
        if speed_diff > 15:
            return "同向速度差冲突"
        elif speed_diff > 5:
            return "同向跟随冲突"
        else:
            return "同向并行冲突"
    elif 0 <= angle < 30 or 330 < angle <= 360:
        if speed1 > 20 or speed2 > 20:
            return "高速直角交叉冲突"
        else:
            return "直角交叉冲突"
    elif 30 <= angle < 70 or 290 < angle <= 330:
        return "斜向交叉冲突"
    elif 110 < angle < 160:
        if speed1 > 25 or speed2 > 25:
            return "高速左转冲突"
        else:
            return "左转冲突"
    elif 200 < angle < 250:
        if speed1 > 25 or speed2 > 25:
            return "高速右转冲突"
        else:
            return "右转冲突"
    elif 250 < angle < 290:
        return "抢道冲突"
    elif speed1 > 40 or speed2 > 40:
        return "超速冲突"
    elif speed1 < 5 or speed2 < 5:
        return "低速干扰冲突"
    else:
        return "机非交互冲突"


# ==============================================
# 🔥 任务1：无人机视频 → 机非轨迹提取（优化版）
# ==============================================
def extract_trajectories():
    # 加载模型并查看实际类别
    if not os.path.exists(CONFIG["model_path"]):
        raise FileNotFoundError("模型文件不存在！")

    model = YOLO(CONFIG["model_path"])
    print(f"✅ 模型加载成功！实际类别: {model.names}")
    print(f"请确认 VEHICLE_CLASSES 和 NON_VEHICLE_CLASSES 与上述类别匹配！")

    # 打开视频
    cap = cv2.VideoCapture(CONFIG["video_input"])
    if not cap.isOpened():
        raise FileNotFoundError(f"视频文件不存在: {CONFIG['video_input']}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ========== 视频预处理优化 ==========
    # 1. 降低视频分辨率
    target_width = min(width, CONFIG["target_width"])
    scale = target_width / width
    target_height = int(height * scale)
    
    # 2. 跳帧处理
    frame_skip = CONFIG["frame_skip"]
    output_fps = fps / frame_skip
    
    print(f"原始分辨率: {width}x{height}, 目标分辨率: {target_width}x{target_height}")
    print(f"原始帧率: {fps}, 输出帧率: {output_fps}")
    print(f"总帧数: {total_frames}, 处理帧数: {total_frames//frame_skip}")
    # ====================================

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(CONFIG["video_output"], fourcc, output_fps, (target_width, target_height))

    frame_idx = 0
    process_count = 0
    print(f"✅ 开始处理视频... 总帧数: {total_frames}, FPS: {fps}")
    
    # 初始化交通流量分析器
    flow_analyzer = TrafficFlowAnalyzer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # 跳帧处理
        if frame_idx % frame_skip != 0:
            continue
        
        # 降低分辨率
        resized_frame = cv2.resize(frame, (target_width, target_height))
        annotated_frame = resized_frame.copy()

        # YOLO多目标跟踪
        results = model.track(
            resized_frame, persist=True, tracker="bytetrack.yaml",
            conf=CONFIG["conf_threshold"], iou=CONFIG["iou_threshold"], verbose=False
        )

        current_objects = []  # 当前帧的所有目标

        if results[0].boxes.id is not None:
            # 解析跟踪结果
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            # 存储轨迹
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                cls_name = model.names[cls]
                center_x, center_y = (x1 + x2) // 2, int(y2 * 0.9)

                # 保存轨迹点
                current_time = round(frame_idx / fps, 2)
                trajectories[track_id].append({
                    "frame": frame_idx, "time": current_time,
                    "pos": (center_x, center_y), "bbox": (x1, y1, x2, y2),
                    "class": cls_name, "conf": round(float(conf), 2),
                })

                # 更新交通流量统计
                flow_analyzer.update_flow(track_id, cls_name, current_time)
                
                # 计算速度并更新速度统计
                speed = calculate_velocity(trajectories[track_id])
                flow_analyzer.update_speed_stats(cls_name, speed)

                current_objects.append({
                    'id': track_id,
                    'class': cls_name,
                    'x': center_x,
                    'y': center_y,
                    'bbox': (x1, y1, x2, y2)
                })

                # 绘制目标框+ID
                color = track_colors[track_id]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID{track_id}|{cls_name}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 绘制轨迹线
            for tid, track in trajectories.items():
                if len(track) > 1:
                    points = [t['pos'] for t in track]
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i - 1], points[i], track_colors[tid], 2)

            # ==============================================
            # 🔥 任务2：机非冲突识别模型（修复版）
            # ==============================================
            # 两两比较当前帧的所有目标
            for i in range(len(current_objects)):
                for j in range(i + 1, len(current_objects)):
                    obj1 = current_objects[i]
                    obj2 = current_objects[j]

                    # 计算像素距离
                    pixel_dist = math.hypot(obj2['x'] - obj1['x'], obj2['y'] - obj1['y'])
                    distance = pixel2meter(pixel_dist)

                    # 计算角度
                    angle = int(math.degrees(math.atan2(abs(obj2['y'] - obj1['y']), abs(obj2['x'] - obj1['x']))))

                    # 获取轨迹
                    track1 = trajectories[obj1['id']]
                    track2 = trajectories[obj2['id']]

                    # 检查轨迹长度，确保有足够的点进行计算
                    if len(track1) < 2 or len(track2) < 2:
                        continue

                    # 判定冲突类型（必须有机非交互）
                    conflict_type = get_conflict_type(obj1['class'], obj2['class'], angle, track1, track2)

                    # 如果没有机非交互，跳过
                    if conflict_type == "非机非冲突":
                        continue

                    # 优化速度计算，使用最近3帧的平均速度
                    def get_avg_speed(track, window=3):
                        if len(track) < window:
                            return calculate_velocity(track) / 3.6
                        speeds = []
                        for i in range(1, min(window, len(track))):
                            pos1 = track[-i-1]['pos']
                            pos2 = track[-i]['pos']
                            time1 = track[-i-1]['time']
                            time2 = track[-i]['time']
                            time_diff = time2 - time1
                            if time_diff > 0:
                                dx = pixel2meter(abs(pos2[0] - pos1[0]))
                                dy = pixel2meter(abs(pos2[1] - pos1[1]))
                                dist = math.hypot(dx, dy)
                                speed = dist / time_diff
                                speeds.append(speed)
                        return sum(speeds) / len(speeds) if speeds else 0

                    speed1 = get_avg_speed(track1)
                    speed2 = get_avg_speed(track2)
                    avg_speed = (speed1 + speed2) / 2

                    # 优化TTC计算，考虑相对速度
                    relative_speed = math.hypot(speed1, speed2) if angle < 90 else abs(speed1 - speed2)
                    ttc = distance / relative_speed if relative_speed > 0 else 999

                    # 优化PET计算，使用更精确的时间差计算
                    def calculate_pet_optimized(track1, track2):
                        try:
                            # 计算两个目标的移动方向
                            def get_direction(track):
                                if len(track) < 2:
                                    return (0, 0)
                                pos1 = track[-2]['pos']
                                pos2 = track[-1]['pos']
                                return (pos2[0] - pos1[0], pos2[1] - pos1[1])
                            
                            dir1 = get_direction(track1)
                            dir2 = get_direction(track2)
                            
                            # 计算方向相似度
                            dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
                            mag1 = math.hypot(dir1[0], dir1[1])
                            mag2 = math.hypot(dir2[0], dir2[1])
                            if mag1 > 0 and mag2 > 0:
                                direction_similarity = dot_product / (mag1 * mag2)
                            else:
                                direction_similarity = 0
                            
                            # 计算距离
                            pos1 = track1[-1]['pos']
                            pos2 = track2[-1]['pos']
                            dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                            
                            # 计算时间差
                            time1 = track1[-1]['time']
                            time2 = track2[-1]['time']
                            time_diff = abs(time1 - time2)
                            
                            # 根据方向和距离调整PET
                            if dist < 150:
                                # 方向越相似，PET越小
                                pet = time_diff * (1 - abs(direction_similarity))
                                return round(pet, 2)
                            else:
                                return 999
                        except:
                            return 999

                    pet = calculate_pet_optimized(track1, track2)

                    # 优化冲突判定条件，增加动态阈值
                    def get_dynamic_thresholds(speed1, speed2):
                        # 速度越快，阈值越严格
                        base_ttc = CONFIG["TTC_THRESHOLD"]
                        base_pet = CONFIG["PET_THRESHOLD"]
                        base_distance = CONFIG["DISTANCE_THRESHOLD"]
                        
                        max_speed = max(speed1 * 3.6, speed2 * 3.6)  # 转换为km/h
                        
                        if max_speed > 40:
                            return base_ttc * 0.7, base_pet * 0.8, base_distance * 1.2
                        elif max_speed > 25:
                            return base_ttc * 0.85, base_pet * 0.9, base_distance * 1.1
                        else:
                            return base_ttc, base_pet, base_distance

                    dynamic_ttc, dynamic_pet, dynamic_distance = get_dynamic_thresholds(speed1, speed2)

                    # 冲突判定（更严格的条件）
                    is_conflict = (distance < dynamic_distance and
                                   ttc < dynamic_ttc and
                                   pet < dynamic_pet)

                    if is_conflict:
                        # 更严格的高危冲突判定
                        risk_level = "高危" if (distance < 1 and ttc < 0.5) else "一般"

                        conflict_records.append({
                            "frame": frame_idx,
                            "time": round(frame_idx / fps, 2),
                            "id1": obj1['id'],
                            "id2": obj2['id'],
                            "type1": obj1['class'],
                            "type2": obj2['class'],
                            "distance(m)": distance,
                            "TTC(s)": round(ttc, 2),
                            "PET(s)": pet,
                            "conflict_type": conflict_type,
                            "risk_level": risk_level,
                            "x": (obj1['x'] + obj2['x']) // 2,
                            "y": (obj1['y'] + obj2['y']) // 2,
                        })

                        # 冲突可视化（标红+文字）
                        x1, y1, x2, y2 = obj1['bbox']
                        cv2.rectangle(annotated_frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 255), 3)
                        # 使用英文显示冲突类型，避免中文乱码
                        conflict_type_en = {
                            "高速直行-横穿冲突": "High-speed Straight-Cross Conflict",
                            "直行-横穿冲突": "Straight-Cross Conflict",
                            "同向速度差冲突": "Same-direction Speed Difference Conflict",
                            "同向跟随冲突": "Same-direction Following Conflict",
                            "同向并行冲突": "Same-direction Parallel Conflict",
                            "高速直角交叉冲突": "High-speed Right-angle Cross Conflict",
                            "直角交叉冲突": "Right-angle Cross Conflict",
                            "斜向交叉冲突": "Oblique Cross Conflict",
                            "高速左转冲突": "High-speed Left-turn Conflict",
                            "左转冲突": "Left-turn Conflict",
                            "高速右转冲突": "High-speed Right-turn Conflict",
                            "右转冲突": "Right-turn Conflict",
                            "抢道冲突": "Lane-changing Conflict",
                            "超速冲突": "Speeding Conflict",
                            "低速干扰冲突": "Low-speed Interference Conflict",
                            "机非交互冲突": "Vehicle-NonVehicle Conflict"
                        }.get(conflict_type, "Unknown Conflict")
                        cv2.putText(annotated_frame, f"CONFLICT:{conflict_type_en}",
                                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 显示当前帧信息
        info_text = f"Frame: {process_count}/{total_frames//frame_skip} | Tracks: {len(trajectories)} | Conflicts: {len(conflict_records)}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(annotated_frame)
        process_count += 1

        if process_count % 50 == 0:
            print(
                f"处理帧：{process_count}/{total_frames//frame_skip} | 跟踪目标：{len(trajectories)} | 已识别冲突：{len(conflict_records)}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 保存交通流量数据
    total_flow, speed_stats = flow_analyzer.save_flow_data()
    print(f"\n✅ 视频处理完成！总冲突数：{len(conflict_records)}")
    print(f"交通流量统计：机动车 {total_flow['vehicle']} 辆，非机动车 {total_flow['non_vehicle']} 辆，总计 {total_flow['total']} 辆")


# ==============================================
# 数据保存函数（修复：确保列名存在）
# ==============================================
def save_trajectories():
    """保存机非轨迹数据"""
    if not trajectories:
        print("⚠️ 无轨迹数据")
        return

    data = []
    for tid, track in trajectories.items():
        for p in track:
            data.append({
                "track_id": tid,
                "frame": p["frame"],
                "time(s)": p["time"],
                "x(px)": p["pos"][0],
                "y(px)": p["pos"][1],
                "class": p["class"],
                "conf": p["conf"]
            })
    df = pd.DataFrame(data)
    df.to_csv(CONFIG["traj_csv"], index=False, encoding='utf-8')
    print(f"✅ 轨迹已保存：{CONFIG['traj_csv']} ({len(df)}条记录)")


def save_conflicts():
    """保存冲突数据（修复空数据问题）"""
    # 确保有列名，即使数据为空
    columns = ["frame", "time", "id1", "id2", "type1", "type2",
               "distance(m)", "TTC(s)", "PET(s)", "conflict_type", "risk_level", "x", "y"]

    if conflict_records:
        df = pd.DataFrame(conflict_records)
    else:
        # 创建空DataFrame，但保留列名
        df = pd.DataFrame(columns=columns)
        print("⚠️ 未检测到冲突，保存空表")

    df.to_csv(CONFIG["conflict_csv"], index=False, encoding='utf-8')
    print(f"✅ 冲突数据已保存：{CONFIG['conflict_csv']}")


# ==============================================
# 🔥 任务3：时空分布分析 + 交叉口安全评价（修复版）
# ==============================================
def analyze_spatial_temporal():
    """冲突热力图分析"""
    if not conflict_records:
        print("⚠️ 无冲突数据，跳过热力图分析")
        return

    df = pd.DataFrame(conflict_records)

    # 从conflict_records直接获取坐标（已保存）
    x = df['x'].tolist()
    y = df['y'].tolist()

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c='red', s=100, alpha=0.6, edgecolors='black')
    plt.title('交叉口冲突热力图（红色=危险区域）')
    plt.xlabel('X坐标（像素）')
    plt.ylabel('Y坐标（像素）')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.savefig(CONFIG["heatmap_path"], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 冲突热力图已保存：{CONFIG['heatmap_path']}")


def safety_evaluation():
    """交叉口交通安全量化评价（修复空数据问题）"""
    # 直接读取CSV而不是使用全局变量
    try:
        df = pd.read_csv(CONFIG["conflict_csv"])
    except:
        df = pd.DataFrame()

    total_conflicts = len(df)

    if total_conflicts == 0:
        # 无冲突的情况
        score = 100
        level = "优秀"
        high_risk = 0
        conflict_types = {}
        avg_distance = 0
        avg_ttc = 0
        avg_pet = 0
        # 新增指标
        flow_density = 0
        speed_distribution = "均匀"
        conflict_severity = "无"
        time_distribution = "均匀"
        spatial_distribution = "均匀"
    else:
        # 统计高危冲突次数
        high_risk = len(df[df['risk_level'] == '高危']) if 'risk_level' in df.columns else 0
        
        # 统计冲突类型分布
        conflict_types = df['conflict_type'].value_counts().to_dict() if 'conflict_type' in df.columns else {}
        
        # 计算平均距离、TTC和PET
        avg_distance = df['distance(m)'].mean() if 'distance(m)' in df.columns else 0
        avg_ttc = df['TTC(s)'].mean() if 'TTC(s)' in df.columns else 0
        avg_pet = df['PET(s)'].mean() if 'PET(s)' in df.columns else 0

        # 新增评价指标
        # 1. 冲突严重程度
        if high_risk / total_conflicts > 0.5:
            conflict_severity = "严重"
        elif high_risk / total_conflicts > 0.2:
            conflict_severity = "中等"
        else:
            conflict_severity = "轻微"
        
        # 2. 时间分布
        if 'time' in df.columns:
            time_values = df['time'].tolist()
            time_bins = [0, 60, 120, 180, 240, 300]
            time_counts = pd.cut(time_values, bins=time_bins, right=False).value_counts()
            if time_counts.max() / time_counts.sum() > 0.6:
                time_distribution = "集中"
            else:
                time_distribution = "均匀"
        else:
            time_distribution = "均匀"
        
        # 3. 空间分布
        if 'x' in df.columns and 'y' in df.columns:
            x_values = df['x'].tolist()
            y_values = df['y'].tolist()
            if len(x_values) > 0:
                x_std = np.std(x_values)
                y_std = np.std(y_values)
                if x_std < 100 and y_std < 100:
                    spatial_distribution = "集中"
                else:
                    spatial_distribution = "分散"
            else:
                spatial_distribution = "均匀"
        else:
            spatial_distribution = "均匀"
        
        # 4. 流量密度（从交通流量文件读取）
        flow_density = 0
        if os.path.exists('output/traffic_flow.csv'):
            try:
                df_flow = pd.read_csv('output/traffic_flow.csv')
                total_vehicles = df_flow['total_count'].sum()
                if total_vehicles > 0:
                    # 假设视频时长为300秒
                    flow_density = total_vehicles / 5  # 辆/分钟
            except:
                pass
        
        # 5. 速度分布
        speed_distribution = "均匀"
        if os.path.exists('output/speed_stats.csv'):
            try:
                df_speed = pd.read_csv('output/speed_stats.csv')
                if not df_speed.empty:
                    speed_std = df_speed['average_speed'].std()
                    if speed_std > 10:
                        speed_distribution = "不均匀"
            except:
                pass

        # 安全评分（100分制）
        # 基于冲突次数、高危冲突、平均距离、TTC和PET计算
        score = 100
        
        # 冲突次数扣分（调整权重）
        score -= min(total_conflicts, 500) * 0.1  # 最多扣50分
        
        # 高危冲突扣分
        score -= min(high_risk, 100) * 0.2  # 最多扣20分
        
        # 平均距离扣分（距离越近扣分越多）
        if avg_distance < 5:
            score -= (5 - avg_distance) * 1
        
        # TTC扣分（TTC越小扣分越多）
        if avg_ttc < 3:
            score -= (3 - avg_ttc) * 3
        
        # PET扣分（PET越小扣分越多）
        if avg_pet < 1:
            score -= (1 - avg_pet) * 3
        
        # 冲突严重程度扣分
        if conflict_severity == "严重":
            score -= 10
        elif conflict_severity == "中等":
            score -= 5
        
        # 时间分布扣分
        if time_distribution == "集中":
            score -= 5
        
        # 空间分布扣分
        if spatial_distribution == "集中":
            score -= 5
        
        # 流量密度扣分
        if flow_density > 50:
            score -= 10
        elif flow_density > 30:
            score -= 5
        
        # 速度分布扣分
        if speed_distribution == "不均匀":
            score -= 5
        
        score = max(0, score)  # 确保分数不为负
        
        # 确定安全等级
        if score >= 90:
            level = "优秀"
        elif score >= 75:
            level = "良好"
        elif score >= 60:
            level = "一般"
        elif score >= 40:
            level = "较差"
        else:
            level = "危险"

    # 生成评价报告
    report = f"""
==================== 交叉口交通安全评价报告 ====================
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 冲突统计结果：
总冲突次数：{total_conflicts} 次
高危冲突次数：{high_risk} 次
冲突类型分布：{conflict_types}
平均冲突距离：{avg_distance:.2f} 米
平均碰撞时间（TTC）：{avg_ttc:.2f} 秒
平均侵占时间（PET）：{avg_pet:.2f} 秒

� 新增评价指标：
冲突严重程度：{conflict_severity}
时间分布特征：{time_distribution}
空间分布特征：{spatial_distribution}
流量密度：{flow_density:.2f} 辆/分钟
速度分布：{speed_distribution}

�� 安全评分：{score:.1f} 分 | 安全等级：【{level}】

💡 优化建议：
{f'1. 冲突热点区域需增设警示标志/减速带' if total_conflicts > 0 else '1. 当前交通状况良好，保持现有管理措施'}
{f'2. 优先治理「{max(conflict_types, key=conflict_types.get) if conflict_types else "无"}」类型冲突' if conflict_types else '2. 建议定期巡查，预防潜在风险'}
{f'3. 高峰时段加强非机动车管控' if high_risk > 0 else '3. 建议持续监测交通流量变化'}
{f'4. 优化交叉口交通信号配时，减少机非冲突' if total_conflicts > 0 else '4. 建议定期进行交通安全评估'}
{f'5. 加强对非机动车驾驶人的安全教育' if any("非机动车" in str(ct) for ct in conflict_types) else '5. 建议优化交通标志标线'}
{f'6. 加强交通流量管控，缓解高峰期拥堵' if flow_density > 30 else '6. 保持现有交通管理措施'}
{f'7. 对速度不均匀路段进行限速管理' if speed_distribution == "不均匀" else '7. 保持现有速度管理措施'}
==============================================================
"""
    with open(CONFIG["report_path"], 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f"✅ 安全评价报告已保存：{CONFIG['report_path']}")


# ==============================================
# 主程序
# ==============================================
if __name__ == "__main__":
    create_output_dir()
    extract_trajectories()  # 任务1
    save_trajectories()  # 保存轨迹
    save_conflicts()  # 任务2（修复版）
    
    # 保存交通流量数据（在extract_trajectories函数中已经初始化并更新了flow_analyzer）
    # 注意：由于flow_analyzer是在extract_trajectories函数内部定义的，
    # 这里需要修改extract_trajectories函数返回flow_analyzer，或者将其定义为全局变量
    # 为了简化，我们在extract_trajectories函数中直接保存流量数据
    
    analyze_spatial_temporal()  # 任务3
    safety_evaluation()  # 任务3（修复版）
    print("\n🎉 全部任务完成！所有结果保存在 output 文件夹中")