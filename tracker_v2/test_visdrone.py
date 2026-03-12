"""
用VisDrone真实航拍数据测试特征匹配管线

从VisDrone中挑选包含车辆的图片:
1. 从第一张图中选一个车辆作为"目标"
2. 在后续图片中搜索类似车辆
3. 测试匹配准确率和速度
"""

import os
import sys
import time
import cv2
import numpy as np
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_bank import TargetFeatureBank
from target_matcher import TargetMatcher
from detector import Detector
from config import MATCH_THRESHOLD


def parse_visdrone_annotation(ann_path):
    """
    VisDrone标注格式: x,y,w,h,score,category,truncation,occlusion
    category: 0=ignored, 1=pedestrian, 2=people, 3=bicycle,
              4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle,
              9=bus, 10=motor, 11=others
    """
    objects = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])
            cat = int(parts[5])
            if w < 10 or h < 10:
                continue
            objects.append({
                'bbox': (x, y, x+w, y+h),
                'category': cat,
                'score': score,
            })
    return objects


def main():
    visdrone_dir = os.path.expanduser('~/datasets/VisDrone/VisDrone2019-DET-val')
    images_dir = os.path.join(visdrone_dir, 'images')
    anns_dir = os.path.join(visdrone_dir, 'annotations')

    if not os.path.exists(images_dir):
        print(f'VisDrone数据集不存在: {images_dir}')
        print('请先运行下载')
        sys.exit(1)

    image_files = sorted(glob(os.path.join(images_dir, '*.jpg')))
    print(f'VisDrone验证集: {len(image_files)} 张图片')

    # 找一张包含多个车辆的图片作为"目标照片"
    target_img = None
    target_crop = None
    target_bbox = None
    target_file = None

    # 车辆类别: car=4, van=5, truck=6, bus=9
    vehicle_cats = {4, 5, 6, 9}

    for img_file in image_files[:50]:  # 在前50张中找
        ann_file = os.path.join(anns_dir, os.path.basename(img_file).replace('.jpg', '.txt'))
        if not os.path.exists(ann_file):
            continue

        objects = parse_visdrone_annotation(ann_file)
        vehicles = [o for o in objects if o['category'] in vehicle_cats and o['score'] > 0]

        if len(vehicles) >= 3:  # 至少3个车辆
            img = cv2.imread(img_file)
            if img is None:
                continue

            # 选最大的车辆作为目标
            best_v = max(vehicles, key=lambda v: (v['bbox'][2]-v['bbox'][0]) * (v['bbox'][3]-v['bbox'][1]))
            x1, y1, x2, y2 = best_v['bbox']

            # 确保裁剪区域有效
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size > 100:
                target_img = img
                target_crop = crop
                target_bbox = (x1, y1, x2, y2)
                target_file = img_file
                break

    if target_crop is None:
        print('未找到合适的目标车辆')
        sys.exit(1)

    print(f'\n目标选自: {os.path.basename(target_file)}')
    print(f'目标框: {target_bbox}, 大小: {target_crop.shape[1]}x{target_crop.shape[0]}')

    # 保存目标裁剪
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visdrone_target.jpg'), target_crop)

    # 构建特征库(只用这一个目标)
    bank = TargetFeatureBank()
    tmpl = bank._extract_features(target_crop, 'visdrone_target')
    bank.templates.append(tmpl)
    print(f'特征提取完成: {tmpl["size_level"]}')

    matcher = TargetMatcher(bank)

    # 在其他图片中搜索
    print(f'\n在其他图片中搜索该目标...')
    print(f'{"="*60}')

    test_images = image_files[:100]  # 测试100张
    found = 0
    total_ms = 0

    for img_file in test_images:
        if img_file == target_file:
            continue

        img = cv2.imread(img_file)
        if img is None:
            continue

        # 读标注，找所有车辆
        ann_file = os.path.join(anns_dir, os.path.basename(img_file).replace('.jpg', '.txt'))
        objects = parse_visdrone_annotation(ann_file) if os.path.exists(ann_file) else []
        vehicles = [o for o in objects if o['category'] in vehicle_cats]

        if not vehicles:
            continue

        # 对每个车辆做特征匹配
        t0 = time.perf_counter()
        best_score = 0
        best_v = None

        for v in vehicles:
            x1, y1, x2, y2 = v['bbox']
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            if crop.size < 100:
                continue

            score, _ = matcher.match_crop(crop, yolo_conf=0.5)
            if score > best_score:
                best_score = score
                best_v = v

        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        fname = os.path.basename(img_file)
        matched = best_score >= MATCH_THRESHOLD

        if matched:
            found += 1
            print(f'  {fname}: {len(vehicles):2d} vehicles, best_match={best_score:.3f} ({elapsed:.0f}ms) ← MATCH')
        # 只打印匹配到的，减少输出

    avg_ms = total_ms / max(1, len(test_images))
    print(f'\n{"="*60}')
    print(f'结果: {found}/{len(test_images)} 张图片匹配到类似目标')
    print(f'平均匹配耗时: {avg_ms:.1f}ms/帧 ({len(test_images)}张)')
    print(f'注: 不同场景的"相似"车辆匹配到是正常的(颜色/形状接近)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
