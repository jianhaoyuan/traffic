from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import math
from datetime import datetime
import logging
import traceback
# 确保matplotlib能正确显示中文
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Arial"]  # 优先使用中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
import tempfile
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 交通冲突分析配置 ==========
# 机非分类定义（必须与你的模型实际类别一致！）
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'vehicle', 'motor']
NON_VEHICLE_CLASSES = ['motorcycle', 'bicycle', 'bike', 'person', 'non_motor']

# 冲突判定参数
TTC_THRESHOLD = 1.5
PET_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 2.0
PIXEL_PER_METER = 30
MAX_TRAJECTORY_LEN = 60

# 工具函数
def pixel2meter(pixel):
    """像素转米"""
    return round(pixel / PIXEL_PER_METER, 2)

def calculate_velocity(track):
    """从轨迹计算速度（km/h）"""
    if len(track) < 2:
        return 0
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

def calculate_pet(track1, track2):
    """计算PET（侵占时间）"""
    try:
        if len(track1) < 2 or len(track2) < 2:
            return 999
        pos1 = track1[-1]['pos']
        pos2 = track2[-1]['pos']
        time1 = track1[-1]['time']
        time2 = track2[-1]['time']
        dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        if dist < 100:
            pet = abs(time1 - time2)
            return round(pet, 2)
        else:
            return 999
    except:
        return 999

def get_conflict_type(obj1_cls, obj2_cls, angle, track1, track2):
    """判定机非冲突类型"""
    # 直接使用类别名称进行判断，因为模型只有motor和non_motor两个类别
    is_vehicle_1 = obj1_cls.lower() == 'motor'
    is_vehicle_2 = obj2_cls.lower() == 'motor'
    is_non_vehicle_1 = obj1_cls.lower() == 'non_motor'
    is_non_vehicle_2 = obj2_cls.lower() == 'non_motor'

    if not ((is_vehicle_1 and is_non_vehicle_2) or (is_non_vehicle_1 and is_vehicle_2)):
        return None

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

def save_conflict_analysis(trajectories, conflict_records, output_dir='output'):
    """保存冲突分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存轨迹数据
    traj_data = []
    for tid, track in trajectories.items():
        for p in track:
            traj_data.append({
                "track_id": tid,
                "frame": p["frame"],
                "time(s)": p["time"],
                "x(px)": p["pos"][0],
                "y(px)": p["pos"][1],
                "class": p["class"],
                "conf": p["conf"]
            })
    df_traj = pd.DataFrame(traj_data)
    df_traj.to_csv(os.path.join(output_dir, 'trajectories.csv'), index=False, encoding='utf-8')
    
    # 保存冲突数据
    if conflict_records:
        df_conflict = pd.DataFrame(conflict_records)
    else:
        df_conflict = pd.DataFrame(columns=["frame", "time", "id1", "id2", "type1", "type2",
                                            "distance(m)", "TTC(s)", "PET(s)", "conflict_type", "risk_level", "x", "y"])
    df_conflict.to_csv(os.path.join(output_dir, 'conflicts.csv'), index=False, encoding='utf-8')
    
    # 生成热力图
    if conflict_records:
        df = pd.DataFrame(conflict_records)
        x = df['x'].tolist()
        y = df['y'].tolist()
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, c='red', s=100, alpha=0.6, edgecolors='black')
        plt.title('交叉口冲突热力图（红色=危险区域）')
        plt.xlabel('X坐标（像素）')
        plt.ylabel('Y坐标（像素）')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'conflict_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 生成安全评价报告
    total_conflicts = len(conflict_records)
    if total_conflicts == 0:
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
        df = pd.DataFrame(conflict_records)
        high_risk = len(df[df['risk_level'] == '高危'])
        conflict_types = df['conflict_type'].value_counts().to_dict()
        avg_distance = df['distance(m)'].mean()
        avg_ttc = df['TTC(s)'].mean()
        avg_pet = df['PET(s)'].mean()
        
        # 新增评价指标
        # 1. 冲突严重程度
        if high_risk / total_conflicts > 0.5:
            conflict_severity = "严重"
        elif high_risk / total_conflicts > 0.2:
            conflict_severity = "中等"
        else:
            conflict_severity = "轻微"
        
        # 2. 时间分布
        time_values = df['time'].tolist()
        time_bins = [0, 60, 120, 180, 240, 300]
        time_counts = pd.cut(time_values, bins=time_bins, right=False).value_counts()
        if time_counts.max() / time_counts.sum() > 0.6:
            time_distribution = "集中"
        else:
            time_distribution = "均匀"
        
        # 3. 空间分布
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
        
        # 4. 流量密度（从交通流量文件读取）
        flow_density = 0
        if os.path.exists(os.path.join(output_dir, 'traffic_flow.csv')):
            try:
                df_flow = pd.read_csv(os.path.join(output_dir, 'traffic_flow.csv'))
                total_vehicles = df_flow['total_count'].sum()
                if total_vehicles > 0:
                    # 假设视频时长为300秒
                    flow_density = total_vehicles / 5  # 辆/分钟
            except:
                pass
        
        # 5. 速度分布
        speed_distribution = "均匀"
        if os.path.exists(os.path.join(output_dir, 'speed_stats.csv')):
            try:
                df_speed = pd.read_csv(os.path.join(output_dir, 'speed_stats.csv'))
                if not df_speed.empty:
                    speed_std = df_speed['average_speed'].std()
                    if speed_std > 10:
                        speed_distribution = "不均匀"
            except:
                pass
        
        # 计算综合评分
        score = 100
        # 冲突次数扣分
        score -= min(total_conflicts, 500) * 0.1
        # 高危冲突扣分
        score -= min(high_risk, 100) * 0.2
        # 距离扣分
        if avg_distance < 5:
            score -= (5 - avg_distance) * 1
        # TTC扣分
        if avg_ttc < 3:
            score -= (3 - avg_ttc) * 3
        # PET扣分
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
        
        score = max(0, score)
        
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

📈 新增评价指标：
冲突严重程度：{conflict_severity}
时间分布特征：{time_distribution}
空间分布特征：{spatial_distribution}
流量密度：{flow_density:.2f} 辆/分钟
速度分布：{speed_distribution}

🚦 安全评分：{score:.1f} 分 | 安全等级：【{level}】

💡 优化建议：
{f'1. 冲突热点区域需增设警示标志/减速带' if total_conflicts > 0 else '1. 当前交通状况良好，保持现有管理措施'}
{f'2. 优先治理「{max(conflict_types, key=conflict_types.get) if conflict_types else "无"}」类型冲突' if conflict_types else '2. 建议定期巡查，预防潜在风险'}
{f'3. 高峰时段加强非机动车管控' if high_risk > 0 else '3. 建议持续监测交通流量变化'}
{f'4. 优化交叉口交通信号配时，减少机非冲突' if total_conflicts > 0 else '4. 建议定期进行交通安全评估'}
{f'5. 建议优化交通标志标线' if total_conflicts > 0 else '5. 建议优化交通标志标线'}
{f'6. 加强交通流量管控，缓解高峰期拥堵' if flow_density > 30 else '6. 保持现有交通管理措施'}
{f'7. 对速度不均匀路段进行限速管理' if speed_distribution == "不均匀" else '7. 保持现有速度管理措施'}
==============================================================
"""
    with open(os.path.join(output_dir, 'safety_evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    return total_conflicts, high_risk, score, level
# =========================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 加载模型
model = YOLO('runs/detect/train/weights/best.pt')
# 打印模型类别
print(f"模型类别: {model.names}")

# 主页
@app.route('/')
def index():
    return render_template('index.html')

# 图片检测页面
@app.route('/image_detection', methods=['GET', 'POST'])
def image_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # 保存上传的图片
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # 进行目标检测
            results = model(filename)
            
            # 保存检测结果
            output_filename = f"detected_{file.filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            results[0].save(output_path)
            
            return render_template('image_detection.html', original=file.filename, detected=output_filename)
    return render_template('image_detection.html')

# 视频检测页面（优化版）
@app.route('/video_detection', methods=['GET', 'POST'])
def video_detection():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                logger.warning('No file uploaded')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                logger.warning('Empty filename')
                return redirect(request.url)
            if file:
                # 保存上传的视频
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                try:
                    file.save(filename)
                    logger.info(f'Uploaded file saved: {filename}')
                except Exception as e:
                    logger.error(f'Error saving file: {str(e)}')
                    return render_template('video_detection.html', error=f"保存文件失败: {str(e)}")
            
            # 直接处理视频并保存结果
            cap = cv2.VideoCapture(filename)
            if not cap.isOpened():
                logger.error(f'Cannot open video file: {filename}')
                return render_template('video_detection.html', error="无法打开视频文件")
            
            # 获取视频参数
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f'Video parameters: {width}x{height}, {fps} FPS, {total_frames} frames')
            
            # ========== 视频预处理优化 ==========
            # 1. 降低视频分辨率（最大宽度960，提高速度）
            target_width = min(width, 960)
            scale = target_width / width
            target_height = int(height * scale)
            
            # 2. 跳帧处理（每3帧处理1帧，提高速度）
            frame_skip = 3
            output_fps = fps / frame_skip
            
            logger.info(f'Processing parameters: target resolution={target_width}x{target_height}, frame skip={frame_skip}, output FPS={output_fps}')
            # ====================================
            
            # 准备输出视频（使用H.264编码，确保浏览器兼容）
            output_filename = f"detected_{os.path.splitext(file.filename)[0]}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # 尝试使用H.264编码
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (target_width, target_height))
                if not out.isOpened():
                    raise Exception("H.264编码器不可用")
                logger.info('Using H.264 encoder')
            except Exception as e:
                logger.warning(f'H.264 encoder failed: {str(e)}. Trying mp4v encoder.')
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 备用编码
                    out = cv2.VideoWriter(output_path, fourcc, output_fps, (target_width, target_height))
                    if not out.isOpened():
                        raise Exception("mp4v编码器不可用")
                    logger.info('Using mp4v encoder')
                except Exception as e:
                    logger.error(f'Video writer initialization failed: {str(e)}')
                    return render_template('video_detection.html', error=f"视频编码器初始化失败: {str(e)}")
            
            # ========== 初始化冲突分析变量 ==========
            trajectories = defaultdict(lambda: deque(maxlen=MAX_TRAJECTORY_LEN))
            conflict_records = []
            np.random.seed(42)
            track_colors = defaultdict(lambda: tuple(map(int, np.random.randint(50, 255, 3))))
            # 初始化交通流量分析器
            flow_analyzer = TrafficFlowAnalyzer()
            # ========================================
            
            # 处理每一帧
            frame_count = 0
            process_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 跳帧处理
                if frame_count % frame_skip != 0:
                    continue
                
                # 降低分辨率
                resized_frame = cv2.resize(frame, (target_width, target_height))
                annotated_frame = resized_frame.copy()
                
                # 进行目标检测和跟踪
                try:
                    results = model.track(resized_frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.5)
                except Exception as e:
                    logger.error(f'Error during tracking: {str(e)}')
                    continue
                
                current_objects = []
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        cls_name = model.names[cls]
                        center_x, center_y = (x1 + x2) // 2, int(y2 * 0.9)
                        
                        # 保存轨迹点
                        current_time = round(process_count / output_fps, 2)
                        trajectories[track_id].append({
                            "frame": process_count,
                            "time": current_time,
                            "pos": (center_x, center_y),
                            "bbox": (x1, y1, x2, y2),
                            "class": cls_name,
                            "conf": round(float(conf), 2),
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
                    
                    # ========== 冲突识别 ==========
                    for i in range(len(current_objects)):
                        for j in range(i + 1, len(current_objects)):
                            obj1 = current_objects[i]
                            obj2 = current_objects[j]
                            
                            pixel_dist = math.hypot(obj2['x'] - obj1['x'], obj2['y'] - obj1['y'])
                            distance = pixel2meter(pixel_dist)
                            angle = int(math.degrees(math.atan2(abs(obj2['y'] - obj1['y']), abs(obj2['x'] - obj1['x']))))
                            
                            track1 = trajectories[obj1['id']]
                            track2 = trajectories[obj2['id']]
                            
                            # 检查轨迹长度，确保有足够的点进行计算
                            if len(track1) < 2 or len(track2) < 2:
                                continue
                            
                            conflict_type = get_conflict_type(obj1['class'], obj2['class'], angle, track1, track2)
                            
                            if conflict_type is None:
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
                                base_ttc = TTC_THRESHOLD
                                base_pet = PET_THRESHOLD
                                base_distance = DISTANCE_THRESHOLD
                                
                                max_speed = max(speed1 * 3.6, speed2 * 3.6)  # 转换为km/h
                                
                                if max_speed > 40:
                                    return base_ttc * 0.7, base_pet * 0.8, base_distance * 1.2
                                elif max_speed > 25:
                                    return base_ttc * 0.85, base_pet * 0.9, base_distance * 1.1
                                else:
                                    return base_ttc, base_pet, base_distance
                            
                            dynamic_ttc, dynamic_pet, dynamic_distance = get_dynamic_thresholds(speed1, speed2)
                            
                            is_conflict = (distance < dynamic_distance and
                                          ttc < dynamic_ttc and
                                          pet < dynamic_pet)
                            
                            if is_conflict:
                                risk_level = "高危" if (distance < 1 and ttc < 0.5) else "一般"
                                
                                conflict_records.append({
                                    "frame": process_count,
                                    "time": round(process_count / output_fps, 2),
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
                                
                                # 冲突可视化
                                x1, y1, x2, y2 = obj1['bbox']
                                cv2.rectangle(annotated_frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (0, 0, 255), 3)
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
                    # =================================
                
                # 显示当前帧信息
                info_text = f"Frame: {process_count}/{total_frames//frame_skip} | Tracks: {len(trajectories)} | Conflicts: {len(conflict_records)}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 写入输出视频
                out.write(annotated_frame)
                process_count += 1
                
                # 打印进度
                if process_count % 10 == 0:
                    logger.info(f'Processing progress: {process_count}/{total_frames//frame_skip}')
            
            # 释放资源
            cap.release()
            out.release()
            
            # ========== 保存冲突分析结果 ==========
            logger.info(f'Video processing completed! Total conflicts: {len(conflict_records)}')
            
            # 保存交通流量数据
            try:
                total_flow, speed_stats = flow_analyzer.save_flow_data()
                logger.info(f'Traffic flow: vehicles={total_flow["vehicle"]}, non-vehicles={total_flow["non_vehicle"]}, total={total_flow["total"]}')
            except Exception as e:
                logger.error(f'Error saving flow data: {str(e)}')
            
            # 保存冲突分析数据
            try:
                total_conflicts, high_risk, score, level = save_conflict_analysis(trajectories, conflict_records)
                logger.info(f'Safety score: {score:.1f} | Safety level: {level}')
            except Exception as e:
                logger.error(f'Error saving conflict analysis: {str(e)}')
            # =====================================
            
            # 检查输出文件是否存在
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f'Video detection successful, output file: {output_path}, size: {os.path.getsize(output_path)} bytes')
                return render_template('video_detection.html', original=file.filename, detected=output_filename)
            else:
                logger.error(f'Video detection failed, output file: {output_path}, exists: {os.path.exists(output_path)}, size: {os.path.getsize(output_path) if os.path.exists(output_path) else 0} bytes')
                import glob
                outputs_files = glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], '*'))
                logger.info(f'Outputs directory contents: {outputs_files}')
                return render_template('video_detection.html', error="视频检测失败，输出文件未生成")
    except Exception as e:
        logger.error(f'Error in video_detection: {str(e)}\n{traceback.format_exc()}')
        return render_template('video_detection.html', error=f"处理过程中发生错误: {str(e)}")
    return render_template('video_detection.html')

# 交通冲突信息页面（默认）
@app.route('/conflict_info')
def conflict_info():
    # 读取冲突数据
    conflict_csv = 'output/conflicts.csv'
    trajectories_csv = 'output/trajectories.csv'
    report_txt = 'output/safety_evaluation.txt'
    
    conflicts = []
    trajectories = []
    report = ""
    
    # 读取冲突数据
    if os.path.exists(conflict_csv):
        df = pd.read_csv(conflict_csv)
        conflicts = df.to_dict('records')
    
    # 读取轨迹数据
    if os.path.exists(trajectories_csv):
        df = pd.read_csv(trajectories_csv)
        trajectories = df.to_dict('records')
    
    # 读取安全评价报告
    if os.path.exists(report_txt):
        with open(report_txt, 'r', encoding='utf-8') as f:
            report = f.read()
    
    # 检查热力图是否存在
    heatmap_path = 'output/conflict_heatmap.png'
    heatmap_exists = os.path.exists(heatmap_path)
    
    return render_template('conflict_info.html', conflicts=conflicts, trajectories=trajectories, report=report, heatmap_exists=heatmap_exists, video_filename='default')

# 交通冲突信息页面（视频检测的下一级）
@app.route('/video_detection/conflict_info/<video_filename>')
def conflict_info_with_video(video_filename):
    # 读取冲突数据
    conflict_csv = 'output/conflicts.csv'
    trajectories_csv = 'output/trajectories.csv'
    report_txt = 'output/safety_evaluation.txt'
    
    conflicts = []
    trajectories = []
    report = ""
    
    # 读取冲突数据
    if os.path.exists(conflict_csv):
        df = pd.read_csv(conflict_csv)
        conflicts = df.to_dict('records')
    
    # 读取轨迹数据
    if os.path.exists(trajectories_csv):
        df = pd.read_csv(trajectories_csv)
        trajectories = df.to_dict('records')
    
    # 读取安全评价报告
    if os.path.exists(report_txt):
        with open(report_txt, 'r', encoding='utf-8') as f:
            report = f.read()
    
    # 检查热力图是否存在
    heatmap_path = 'output/conflict_heatmap.png'
    heatmap_exists = os.path.exists(heatmap_path)
    
    return render_template('conflict_info.html', conflicts=conflicts, trajectories=trajectories, report=report, heatmap_exists=heatmap_exists, video_filename=video_filename)

# 提供静态文件访问
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# 提供output文件夹中的文件访问
@app.route('/output/<path:path>')
def send_output(path):
    return send_from_directory('output', path)

# 提供直接访问outputs目录的路由
@app.route('/outputs/<path:path>')
def send_outputs(path):
    return send_from_directory('static/outputs', path)

# 移除了导出功能

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)