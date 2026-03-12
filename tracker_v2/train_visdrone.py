"""
用VisDrone数据集训练YOLO — 航拍目标检测

VisDrone会自动下载(~2.5GB)
10类: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

训练完后导出best.pt用于V2.0系统
"""

import os
import shutil
from ultralytics import YOLO


def main():
    print('='*50)
    print('  VisDrone YOLO训练')
    print('  数据集将自动下载(首次约2.5GB)')
    print('='*50)

    # 使用YOLOv8n (最轻量，适合Jetson)
    model = YOLO('yolov8n.pt')

    # 训练 — VisDrone数据集会自动下载
    model.train(
        data='VisDrone.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device='cpu',            # 开发机CPU训练; Jetson上改为 device=0
        name='visdrone_v2',
        patience=15,
        # 数据增强
        degrees=15,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        mixup=0.1,
        # 小目标优化
        copy_paste=0.1,
    )

    # 复制best.pt
    best_pt = model.trainer.best
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_visdrone.pt')
    shutil.copy2(str(best_pt), dst)
    print(f'\n训练完成! 模型保存到: {dst}')

    # 导出ONNX
    best_model = YOLO(best_pt)
    best_model.export(format='onnx', imgsz=640, simplify=True)
    onnx_path = str(best_pt).replace('.pt', '.onnx')
    if os.path.exists(onnx_path):
        dst_onnx = dst.replace('.pt', '.onnx')
        shutil.copy2(onnx_path, dst_onnx)
        print(f'ONNX已导出: {dst_onnx}')

    print('\n使用方法:')
    print('  python main.py --model best_visdrone.pt --photos target_photos/')


if __name__ == '__main__':
    main()
