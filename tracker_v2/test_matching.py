"""
离线特征匹配测试 — 不需要摄像头，用图片验证特征提取+匹配管线

用法:
  1. 准备测试照片集:
     target_photos/
     ├── far_01.jpg       # 目标远距离照片
     ├── mid_01.jpg       # 中距离
     ├── near_01.jpg      # 近距离
     └── target_info.json # 标注信息

  2. 准备测试帧(模拟摄像头画面):
     test_frames/
     ├── frame_001.jpg
     ├── frame_002.jpg
     └── ...

  3. 运行: python test_matching.py --photos target_photos/ --frames test_frames/

也可以用单张图片快速测试:
  python test_matching.py --photos target_photos/ --test_image some_frame.jpg
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from glob import glob

from config import *
from feature_bank import TargetFeatureBank
from target_matcher import TargetMatcher
from detector import Detector


def test_feature_extraction(photos_dir):
    """测试特征提取"""
    print('\n' + '='*50)
    print('测试1: 特征提取')
    print('='*50)

    bank = TargetFeatureBank()
    t0 = time.time()
    count = bank.load_from_dir(photos_dir)
    elapsed = time.time() - t0

    print(f'  加载 {count} 个模板, 耗时 {elapsed*1000:.1f}ms')

    for i, tmpl in enumerate(bank.templates):
        print(f'  模板[{i}] {tmpl["name"]}: '
              f'size_level={tmpl["size_level"]}, '
              f'hist_shape={tmpl["histogram"].shape}, '
              f'hu={tmpl["hu_moments"][:3].round(2)}, '
              f'scales={list(tmpl["multi_scale"].keys())}')

    return bank


def test_self_matching(bank):
    """测试模板自匹配（应该得高分）"""
    print('\n' + '='*50)
    print('测试2: 模板自匹配（baseline，分数应>0.7）')
    print('='*50)

    matcher = TargetMatcher(bank)

    for i, tmpl in enumerate(bank.templates):
        score, idx = matcher.match_crop(tmpl['image'], yolo_conf=0.8)
        print(f'  模板[{i}] {tmpl["name"]} 自匹配分: {score:.3f} {"PASS" if score > 0.7 else "FAIL"}')


def test_cross_matching(bank):
    """测试模板间交叉匹配（不同视角的同一目标应得中高分）"""
    print('\n' + '='*50)
    print('测试3: 模板交叉匹配')
    print('='*50)

    matcher = TargetMatcher(bank)

    for i, tmpl_i in enumerate(bank.templates):
        scores = []
        for j, tmpl_j in enumerate(bank.templates):
            if i == j:
                continue
            score, _ = matcher.match_crop(tmpl_j['image'])
            scores.append(score)
        if scores:
            avg = sum(scores) / len(scores)
            print(f'  模板[{i}] {tmpl_i["name"]} vs 其他: avg={avg:.3f} min={min(scores):.3f} max={max(scores):.3f}')


def test_on_frames(bank, frames_dir, detector):
    """在测试帧上跑完整管线"""
    print('\n' + '='*50)
    print('测试4: 测试帧检测+匹配')
    print('='*50)

    matcher = TargetMatcher(bank)
    frame_files = sorted(glob(os.path.join(frames_dir, '*.jpg')))

    if not frame_files:
        print(f'  未找到测试帧: {frames_dir}')
        return

    for fpath in frame_files[:20]:
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        fname = os.path.basename(fpath)
        h, w = frame.shape[:2]

        # YOLO检测
        t0 = time.time()
        detections = detector.detect(frame)
        det_ms = (time.time() - t0) * 1000

        # 特征匹配
        t0 = time.time()
        best_det, score = matcher.match_detections(frame, detections)
        match_ms = (time.time() - t0) * 1000

        # 纯模板匹配
        t0 = time.time()
        tmpl_result = matcher.template_search(frame)
        tmpl_ms = (time.time() - t0) * 1000

        det_str = f'{len(detections)} dets'
        match_str = f'match:{score:.3f}' if best_det else 'no match'
        tmpl_str = f'tmpl:{tmpl_result[4]:.3f}' if tmpl_result else 'no tmpl'

        print(f'  {fname}: {det_str}({det_ms:.0f}ms) {match_str}({match_ms:.0f}ms) {tmpl_str}({tmpl_ms:.0f}ms)')

        # 可视化
        vis = frame.copy()
        for det in detections:
            dx1, dy1, dx2, dy2, dconf, _ = det
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (128, 128, 128), 1)

        if best_det:
            bx1, by1, bx2, by2 = best_det[:4]
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(vis, f'MATCH:{score:.2f}', (bx1, by1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if tmpl_result:
            tx1, ty1, tx2, ty2, tscore = tmpl_result
            cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)

        out_path = os.path.join(frames_dir, f'result_{fname}')
        cv2.imwrite(out_path, vis)

    print(f'  结果图已保存到 {frames_dir}/result_*.jpg')


def test_single_image(bank, image_path, detector):
    """单张图片快速测试"""
    print('\n' + '='*50)
    print(f'测试: {image_path}')
    print('='*50)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f'  无法读取: {image_path}')
        return

    matcher = TargetMatcher(bank)
    h, w = frame.shape[:2]

    # YOLO
    detections = detector.detect(frame)
    print(f'  YOLO检测到 {len(detections)} 个目标')

    # 特征匹配
    best_det, score = matcher.match_detections(frame, detections)
    if best_det:
        x1, y1, x2, y2 = best_det[:4]
        br = max((x2-x1)/w, (y2-y1)/h)
        print(f'  最佳匹配: conf={best_det[4]:.2f} match={score:.3f} size={br:.0%}')
    else:
        print(f'  未匹配到目标')

    # 模板匹配
    tmpl = matcher.template_search(frame)
    if tmpl:
        print(f'  模板匹配: score={tmpl[4]:.3f} pos=({tmpl[0]},{tmpl[1]})-({tmpl[2]},{tmpl[3]})')

    # 可视化
    vis = frame.copy()
    for det in detections:
        dx1, dy1, dx2, dy2, dconf, _ = det
        cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (128, 128, 128), 1)
        cv2.putText(vis, f'{dconf:.2f}', (dx1, dy1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    if best_det:
        bx1, by1, bx2, by2 = best_det[:4]
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
        cv2.putText(vis, f'TARGET match:{score:.2f}', (bx1, by1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out = image_path.replace('.jpg', '_result.jpg')
    cv2.imwrite(out, vis)
    print(f'  结果已保存: {out}')


def main():
    parser = argparse.ArgumentParser(description='V2.0 离线特征匹配测试')
    parser.add_argument('--photos', default='target_photos', help='目标照片集目录')
    parser.add_argument('--frames', default=None, help='测试帧目录')
    parser.add_argument('--test_image', default=None, help='单张测试图片')
    parser.add_argument('--model', default='best.pt', help='YOLO模型')
    parser.add_argument('--conf', type=float, default=0.3)
    args = parser.parse_args()

    photos_dir = args.photos
    if not os.path.isabs(photos_dir):
        photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), photos_dir)

    if not os.path.exists(photos_dir):
        print(f'照片集不存在: {photos_dir}')
        print(f'请先准备目标照片集，参见 --help')
        sys.exit(1)

    # 特征提取测试
    bank = test_feature_extraction(photos_dir)
    if not bank.is_loaded():
        print('特征库为空，退出')
        sys.exit(1)

    # 自匹配测试
    test_self_matching(bank)

    # 交叉匹配测试
    test_cross_matching(bank)

    # 如果有YOLO模型，跑检测+匹配测试
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

    detector = None
    if os.path.exists(model_path):
        detector = Detector(model_path=model_path, conf=args.conf)
        detector.load()

        if args.frames:
            test_on_frames(bank, args.frames, detector)

        if args.test_image:
            test_single_image(bank, args.test_image, detector)
    else:
        print(f'\n[跳过] YOLO模型不存在({model_path})，跳过检测测试')

    print('\n' + '='*50)
    print('测试完成!')
    print('='*50)


if __name__ == '__main__':
    main()
