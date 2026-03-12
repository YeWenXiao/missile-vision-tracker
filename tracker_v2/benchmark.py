"""
性能基准测试 — 测量各组件延迟，预估Jetson帧率
"""

import os
import time
import cv2
import numpy as np

from config import *
from feature_bank import TargetFeatureBank
from target_matcher import TargetMatcher
from detector import Detector


def bench(name, func, n=20):
    """运行n次取中位数"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    med = times[len(times)//2]
    avg = sum(times) / len(times)
    print(f'  {name:30s}  med={med:7.1f}ms  avg={avg:7.1f}ms  min={times[0]:7.1f}ms  max={times[-1]:7.1f}ms')
    return med


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    photos_dir = os.path.join(base, 'target_photos')
    frames_dir = os.path.join(base, 'test_frames')

    # 加载
    bank = TargetFeatureBank()
    bank.load_from_dir(photos_dir)
    matcher = TargetMatcher(bank)

    model_path = os.path.join(base, 'best.pt')
    detector = Detector(model_path=model_path, conf=0.3)
    detector.load()

    # 测试帧
    frame = cv2.imread(os.path.join(frames_dir, 'frame_000.jpg'))
    h, w = frame.shape[:2]

    print(f'\n性能基准测试 (帧: {w}x{h})')
    print('='*70)

    # 1. YOLO检测
    # 预热
    for _ in range(3):
        detector.detect(frame)

    det_ms = bench('YOLO检测', lambda: detector.detect(frame))

    # 2. 特征匹配 (对单个crop)
    dets = detector.detect(frame)
    if dets:
        x1, y1, x2, y2, _, _ = dets[0]
        crop = frame[y1:y2, x1:x2]
        match_ms = bench('特征匹配(单crop)', lambda: matcher.match_crop(crop, 0.5))
    else:
        match_ms = 0
        print('  [跳过] 无检测结果')

    # 3. 颜色直方图
    if dets:
        bench('  颜色直方图', lambda: matcher._best_color_match(crop))

    # 4. Hu矩
    if dets:
        bench('  Hu矩匹配', lambda: matcher._best_shape_match(crop))

    # 5. ORB特征
    if dets:
        bench('  ORB纹理匹配', lambda: matcher._best_orb_match(crop))

    # 6. 模板全图搜索
    tmpl_ms = bench('模板全图搜索', lambda: matcher.template_search(frame), n=10)

    # 7. 完整管线 (YOLO + 匹配)
    def full_pipeline():
        d = detector.detect(frame)
        if d:
            matcher.match_detections(frame, d)
    full_ms = bench('完整管线(检测+匹配)', full_pipeline)

    # 汇总
    print('\n' + '='*70)
    print('帧率预估:')
    print(f'  本机(CPU):    {1000/full_ms:.0f} FPS (检测{det_ms:.0f}ms + 匹配{match_ms:.0f}ms)')

    # Jetson Orin Nano TensorRT预估 (YOLO ~15-25ms)
    jetson_yolo = 20  # ms, TensorRT FP16
    jetson_full = jetson_yolo + match_ms
    print(f'  Jetson(预估):  {1000/jetson_full:.0f} FPS (TensorRT {jetson_yolo}ms + 匹配{match_ms:.0f}ms)')
    print(f'  目标: >30 FPS → {"达标" if 1000/jetson_full > 30 else "需要优化"}')
    print('='*70)


if __name__ == '__main__':
    main()
