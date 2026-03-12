"""
从现有dataset制作target_photos和test_frames，用于离线测试
"""

import os
import json
import cv2

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_photos')
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_frames')

os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

images_dir = os.path.join(DATASET_DIR, 'images')
labels_dir = os.path.join(DATASET_DIR, 'labels')

# 找所有有标注的图片
pairs = []
for f in sorted(os.listdir(images_dir)):
    if not f.endswith('.jpg'):
        continue
    lbl = f.replace('.jpg', '.txt')
    lbl_path = os.path.join(labels_dir, lbl)
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        pairs.append(f)

print(f'找到 {len(pairs)} 张有标注的图片')

# 前5张作为target_photos（模拟不同距离的目标照片）
crops_info = {}
photo_names = ['far_01.jpg', 'far_02.jpg', 'mid_01.jpg', 'mid_02.jpg', 'near_01.jpg']

for i, name in enumerate(photo_names):
    if i >= len(pairs):
        break

    src = pairs[i * (len(pairs) // 5)]  # 均匀采样
    img_path = os.path.join(images_dir, src)
    lbl_path = os.path.join(labels_dir, src.replace('.jpg', '.txt'))

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # 读取YOLO标注(归一化格式) → 像素坐标
    with open(lbl_path) as f:
        parts = f.readline().strip().split()
    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    # 稍微扩大裁剪区域
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    # 保存完整图片到target_photos
    dst = os.path.join(PHOTOS_DIR, name)
    cv2.imwrite(dst, img)
    crops_info[name] = [x1, y1, x2, y2]
    print(f'  {name} <- {src} crop=[{x1},{y1},{x2},{y2}]')

# 写target_info.json
info = {
    'description': '从训练数据集自动生成的测试照片集',
    'crops': crops_info,
}
with open(os.path.join(PHOTOS_DIR, 'target_info.json'), 'w') as f:
    json.dump(info, f, indent=2)
print(f'\ntarget_info.json 已生成，{len(crops_info)} 个标注')

# 另外20张作为test_frames（模拟摄像头画面）
count = 0
for i in range(5, min(25, len(pairs))):
    src = pairs[i]
    img = cv2.imread(os.path.join(images_dir, src))
    if img is None:
        continue
    dst = os.path.join(FRAMES_DIR, f'frame_{count:03d}.jpg')
    cv2.imwrite(dst, img)
    count += 1

print(f'\n已生成 {count} 张测试帧到 test_frames/')
print('完成! 现在可以运行: python test_matching.py')
