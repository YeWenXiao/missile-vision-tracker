"""
最简Demo: 连接A8mini RTSP → YOLOv5检测 → 实时显示

用法:
  python demo.py                         # 连接A8mini
  python demo.py --source image.jpg      # 测试图片
  python demo.py --source 0              # 本地摄像头
"""

import argparse
from ultralytics import YOLO
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='rtsp://192.168.144.25:8554/main.264',
                        help='RTSP地址/图片/摄像头编号')
    parser.add_argument('--conf', type=float, default=0.6,
                        help='置信度阈值')
    parser.add_argument('--model', default='best.pt')
    args = parser.parse_args()

    model = YOLO(args.model)
    print(f'模型: {args.model}')
    print(f'输入: {args.source}')
    print(f'置信度阈值: {args.conf}')
    print('按 q 退出\n')

    # ultralytics 直接支持 RTSP/图片/摄像头
    results = model.predict(
        source=args.source,
        conf=args.conf,
        stream=True,        # 流式处理，逐帧返回
        show=True,          # 实时显示窗口
        save=True,          # 保存结果图片
        verbose=False,
    )

    for r in results:
        if r.boxes and len(r.boxes) > 0:
            for box in r.boxes:
                conf = box.conf[0].item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                print(f'目标! conf={conf:.2f} center=({cx:.0f},{cy:.0f})')


if __name__ == '__main__':
    main()
