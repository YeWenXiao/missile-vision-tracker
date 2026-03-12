"""
鲁棒性测试 — 对目标模板做各种变换，验证特征匹配在真实场景下的表现

测试项目:
1. 尺度变化 (模拟远/中/近距离)
2. 旋转 (模拟不同拍摄角度)
3. 亮度变化 (模拟光照变化)
4. 噪声 (模拟传感器噪声)
5. 模糊 (模拟运动/对焦模糊)
6. 部分遮挡
7. 非目标干扰物对比
"""

import os
import sys
import cv2
import numpy as np
import time

from config import MATCH_THRESHOLD, VERIFY_THRESHOLD
from feature_bank import TargetFeatureBank
from target_matcher import TargetMatcher


def augment_scale(img, scales):
    results = {}
    h, w = img.shape[:2]
    for s in scales:
        new = cv2.resize(img, (max(8, int(w*s)), max(8, int(h*s))))
        results[f'scale_{s:.1f}x'] = new
    return results


def augment_rotate(img, angles):
    results = {}
    h, w = img.shape[:2]
    center = (w//2, h//2)
    for a in angles:
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        results[f'rot_{a}deg'] = rotated
    return results


def augment_brightness(img, factors):
    results = {}
    for f in factors:
        adjusted = cv2.convertScaleAbs(img, alpha=f, beta=0)
        results[f'bright_{f:.1f}x'] = adjusted
    return results


def augment_noise(img, stds):
    results = {}
    for std in stds:
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        results[f'noise_std{std}'] = noisy
    return results


def augment_blur(img, kernels):
    results = {}
    for k in kernels:
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        results[f'blur_k{k}'] = blurred
    return results


def augment_occlude(img, ratios):
    results = {}
    h, w = img.shape[:2]
    for r in ratios:
        occluded = img.copy()
        oh = int(h * r)
        ow = int(w * r)
        y = np.random.randint(0, max(1, h - oh))
        x = np.random.randint(0, max(1, w - ow))
        occluded[y:y+oh, x:x+ow] = 128  # 灰色遮挡
        results[f'occlude_{r:.0%}'] = occluded
    return results


def generate_non_targets(frame, target_box, n=5):
    """从帧中随机裁剪非目标区域"""
    h, w = frame.shape[:2]
    tx1, ty1, tx2, ty2 = target_box
    tw, th_box = tx2 - tx1, ty2 - ty1
    results = {}

    for i in range(n):
        for _ in range(50):  # 尝试50次找不重叠的区域
            rx = np.random.randint(0, max(1, w - tw))
            ry = np.random.randint(0, max(1, h - th_box))
            # 检查不与目标重叠
            if (rx > tx2 or rx + tw < tx1 or ry > ty2 or ry + th_box < ty1):
                crop = frame[ry:ry+th_box, rx:rx+tw]
                if crop.size > 0:
                    results[f'non_target_{i}'] = crop
                    break

    return results


def main():
    photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_photos')
    frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_frames')

    # 加载特征库
    bank = TargetFeatureBank()
    bank.load_from_dir(photos_dir)
    if not bank.is_loaded():
        print('特征库为空'); sys.exit(1)

    matcher = TargetMatcher(bank)

    # 取第一个模板的裁剪图作为基准目标
    target_img = bank.templates[0]['image']
    print(f'基准目标: {bank.templates[0]["name"]} ({target_img.shape[1]}x{target_img.shape[0]})')

    all_tests = {}

    # 1. 尺度变化
    all_tests.update(augment_scale(target_img, [0.25, 0.5, 0.75, 1.5, 2.0, 3.0]))

    # 2. 旋转
    all_tests.update(augment_rotate(target_img, [5, 15, 30, 45, 90, 180]))

    # 3. 亮度
    all_tests.update(augment_brightness(target_img, [0.3, 0.5, 0.7, 1.3, 1.5, 2.0]))

    # 4. 噪声
    all_tests.update(augment_noise(target_img, [10, 25, 50, 80]))

    # 5. 模糊
    all_tests.update(augment_blur(target_img, [3, 5, 9, 15]))

    # 6. 遮挡
    all_tests.update(augment_occlude(target_img, [0.1, 0.2, 0.3, 0.5]))

    # 7. 非目标干扰物
    frame_path = os.path.join(frames_dir, 'frame_000.jpg')
    if os.path.exists(frame_path):
        import json
        info_path = os.path.join(photos_dir, 'target_info.json')
        with open(info_path) as f:
            info = json.load(f)
        first_crop = list(info['crops'].values())[0]
        frame = cv2.imread(frame_path)
        non_targets = generate_non_targets(frame, first_crop)
        all_tests.update(non_targets)

    # 运行所有测试
    print(f'\n{"="*60}')
    print(f'鲁棒性测试: {len(all_tests)} 个变体')
    print(f'匹配阈值: MATCH={MATCH_THRESHOLD} VERIFY={VERIFY_THRESHOLD}')
    print(f'{"="*60}')

    categories = {}
    for name, img in sorted(all_tests.items()):
        if img.size == 0:
            continue
        score, _ = matcher.match_crop(img, yolo_conf=0.5)  # 模拟YOLO检测到

        cat = name.split('_')[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, score))

    for cat, results in sorted(categories.items()):
        print(f'\n--- {cat.upper()} ---')
        for name, score in results:
            is_target = not name.startswith('non')
            if is_target:
                status = 'PASS' if score >= VERIFY_THRESHOLD else 'FAIL'
            else:
                status = 'PASS' if score < MATCH_THRESHOLD else 'FALSE_POS'
            bar = '#' * int(score * 30)
            print(f'  {name:25s} score={score:.3f} [{bar:30s}] {status}')

    # 汇总统计
    print(f'\n{"="*60}')
    print('汇总:')
    total_target = 0
    pass_target = 0
    total_non = 0
    pass_non = 0
    for cat, results in categories.items():
        for name, score in results:
            if name.startswith('non'):
                total_non += 1
                if score < MATCH_THRESHOLD:
                    pass_non += 1
            else:
                total_target += 1
                if score >= VERIFY_THRESHOLD:
                    pass_target += 1

    if total_target > 0:
        print(f'  目标变体通过率: {pass_target}/{total_target} ({pass_target/total_target:.0%})')
    if total_non > 0:
        print(f'  非目标排除率:   {pass_non}/{total_non} ({pass_non/total_non:.0%})')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
