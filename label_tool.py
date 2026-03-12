"""
手动标注工具 — 在图片上拖拽画框标注目标

用法: python label_tool.py

操作:
  鼠标拖拽 = 画框选目标
  s = 保存标注，下一张
  d = 跳过（图中没有目标）
  r = 重画
  q = 退出

标注保存为YOLO格式，可直接训练
"""

import cv2
import os
import glob

# 路径
dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# 从3588的dataset复制图片（如果本地没有）
src_dir = r'C:\Users\MS\Desktop\3588\dataset\images'
if os.path.exists(src_dir) and not glob.glob(os.path.join(images_dir, '*.jpg')):
    import shutil
    for f in os.listdir(src_dir):
        if f.endswith('.jpg') and os.path.isfile(os.path.join(src_dir, f)):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(images_dir, f))
    print(f'已从 3588/dataset 复制图片')

# 获取所有图片
all_images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
if not all_images:
    print(f'没有图片! 请把训练图片放到 {images_dir}/')
    exit()

# 标注状态
drawing = False
sx, sy = 0, 0
ex, ey = 0, 0
boxes = []  # 当前图片的所有框 [(x1,y1,x2,y2), ...]


def mouse_callback(event, x, y, flags, param):
    global drawing, sx, sy, ex, ey
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx, sy = x, y
        ex, ey = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        ex, ey = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        if abs(ex - sx) > 10 and abs(ey - sy) > 10:
            bx1 = min(sx, ex)
            by1 = min(sy, ey)
            bx2 = max(sx, ex)
            by2 = max(sy, ey)
            boxes.append((bx1, by1, bx2, by2))


cv2.namedWindow('Label Tool', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Label Tool', mouse_callback)

# 跳过已标注的
labeled = set()
for f in os.listdir(labels_dir):
    if f.endswith('.txt'):
        labeled.add(f.replace('.txt', ''))

idx = 0
# 跳到第一个未标注的
for i, path in enumerate(all_images):
    name = os.path.splitext(os.path.basename(path))[0]
    if name not in labeled:
        idx = i
        break

total = len(all_images)
saved_count = len(labeled)

print(f'共 {total} 张图片, 已标注 {saved_count} 张')
print(f'操作: 拖拽画框 | s=保存 | d=跳过 | r=重画 | q=退出')
print()

while idx < total:
    path = all_images[idx]
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    if img is None:
        idx += 1
        continue

    h, w = img.shape[:2]
    boxes = []

    while True:
        vis = img.copy()

        # 画已确认的框
        for (bx1, by1, bx2, by2) in boxes:
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(vis, 'target', (bx1, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 画正在拖拽的框
        if drawing:
            cv2.rectangle(vis, (sx, sy), (ex, ey), (0, 255, 255), 2)

        # 状态栏
        info = f'[{idx+1}/{total}] {name}.jpg | 框:{len(boxes)} | s=保存 d=跳过 r=重画 q=退出'
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Label Tool', vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            # 保存标注
            label_path = os.path.join(labels_dir, f'{name}.txt')
            with open(label_path, 'w') as f:
                for (bx1, by1, bx2, by2) in boxes:
                    cx = ((bx1 + bx2) / 2) / w
                    cy = ((by1 + by2) / 2) / h
                    bw = (bx2 - bx1) / w
                    bh = (by2 - by1) / h
                    f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            saved_count += 1
            print(f'  已保存 {name}.txt ({len(boxes)} 个框) [{saved_count}]')
            idx += 1
            break

        elif key == ord('d'):
            # 跳过（无目标）,写空标注
            label_path = os.path.join(labels_dir, f'{name}.txt')
            with open(label_path, 'w') as f:
                pass  # 空文件
            print(f'  跳过 {name}.jpg (无目标)')
            idx += 1
            break

        elif key == ord('r'):
            boxes = []

        elif key == ord('q'):
            print(f'\n标注完成! 共保存 {saved_count} 张')
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

# 生成 dataset.yaml
yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(f'path: {dataset_dir}\n')
    f.write('train: images\n')
    f.write('val: images\n')
    f.write('nc: 1\n')
    f.write("names: ['target']\n")

print(f'\n全部标注完成! 共 {saved_count} 张')
print(f'数据集: {dataset_dir}/')
print(f'\n下一步训练:')
print(f'  python train.py')
