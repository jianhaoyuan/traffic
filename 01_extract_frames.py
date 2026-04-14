import cv2
import os
import argparse


def extract_frames_by_time(video_path, output_dir, interval_sec=1.0):
    """
    按时间间隔抽帧

    参数:
        video_path:     视频路径
        output_dir:     输出目录
        interval_sec:   每隔几秒取一帧
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    print(f"帧率: {fps:.1f}，每 {interval_sec} 秒取一帧（即每 {frame_interval} 帧）")

    saved_count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"完成，共保存 {saved_count} 张")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="视频抽帧工具")
    parser.add_argument("--video", type=str, default="input/test_video.mp4", help="视频文件路径")
    parser.add_argument("--output", type=str, default="frames/2", help="输出目录")
    parser.add_argument("--interval", type=float, default=2.0, help="抽帧时间间隔（秒）")
    args = parser.parse_args()
    
    # 运行抽帧
    extract_frames_by_time(
        video_path=args.video,
        output_dir=args.output,
        interval_sec=args.interval
    )
