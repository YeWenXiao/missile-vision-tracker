"""
训练 YOLOv5 — 用手动标注的数据

用法: python train.py
"""

import os
import shutil
from ultralytics import YOLO

dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# 统计有效标注（非空txt文件）
valid = 0
for f in os.listdir(labels_dir):
    if f.endswith('.txt'):
        fpath = os.path.join(labels_dir, f)
        if os.path.getsize(fpath) > 0:
            valid += 1

print(f'有效标注: {valid} 张')
if valid < 10:
    print('标注太少! 请先运行 label_tool.py 标注更多图片')
    exit()

# 分割训练/验证集
train_img = os.path.join(dataset_dir, 'images', 'train')
val_img = os.path.join(dataset_dir, 'images', 'val')
train_lbl = os.path.join(dataset_dir, 'labels', 'train')
val_lbl = os.path.join(dataset_dir, 'labels', 'val')

# 清理旧的分割
for d in [train_img, val_img, train_lbl, val_lbl]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# 只用有标注的图片
pairs = []
for f in sorted(os.listdir(images_dir)):
    if not f.endswith('.jpg') or not os.path.isfile(os.path.join(images_dir, f)):
        continue
    lbl = f.replace('.jpg', '.txt')
    lbl_path = os.path.join(labels_dir, lbl)
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        pairs.append((f, lbl))

split = int(len(pairs) * 0.8)
for i, (img_f, lbl_f) in enumerate(pairs):
    if i < split:
        shutil.copy2(os.path.join(images_dir, img_f), os.path.join(train_img, img_f))
        shutil.copy2(os.path.join(labels_dir, lbl_f), os.path.join(train_lbl, lbl_f))
    else:
        shutil.copy2(os.path.join(images_dir, img_f), os.path.join(val_img, img_f))
        shutil.copy2(os.path.join(labels_dir, lbl_f), os.path.join(val_lbl, lbl_f))

print(f'训练: {split} 张, 验证: {len(pairs)-split} 张')

# 更新yaml
with open(yaml_path, 'w') as f:
    f.write(f'path: {dataset_dir}\n')
    f.write('train: images/train\n')
    f.write('val: images/val\n')
    f.write('nc: 1\n')
    f.write("names: ['target']\n")

# 训练
print('\n开始训练...')
model = YOLO('yolov5nu.pt')
model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=8,
    device='cpu',
    name='target_v2',
    patience=20,
    # 数据增强 — 关键: 旋转+翻转让模型学会各种角度
    degrees=90,        # 随机旋转 ±90度
    fliplr=0.5,        # 水平翻转
    flipud=0.5,        # 垂直翻转
    mosaic=1.0,        # mosaic增强
    scale=0.5,         # 缩放增强
)

# 导出
best_pt = model.trainer.best
print(f'\n训练完成! best.pt: {best_pt}')
shutil.copy2(str(best_pt), os.path.join(os.path.dirname(__file__), 'best.pt'))
print('已复制 best.pt 到当前目录')

# 导出ONNX
best_model = YOLO(best_pt)
best_model.export(format='onnx', imgsz=640, simplify=True)
onnx_path = str(best_pt).replace('.pt', '.onnx')
if os.path.exists(onnx_path):
    shutil.copy2(onnx_path, os.path.join(os.path.dirname(__file__), 'best.onnx'))
    print('已导出 best.onnx')

print('\n测试: python demo.py --source 图片.jpg --conf 0.6')
