"""
VisDrone快速训练 — 仅用验证集做快速验证(数据量小)
正式训练请等train set下载完后用 train_visdrone.py
"""

import os
import shutil
import cv2
from glob import glob
from ultralytics import YOLO


def convert_visdrone_to_yolo(src_images, src_anns, dst_dir, split='train'):
    """将VisDrone标注转为YOLO格式"""
    img_out = os.path.join(dst_dir, 'images', split)
    lbl_out = os.path.join(dst_dir, 'labels', split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # VisDrone类别映射到简化版(只保留车辆)
    # 原始: 4=car, 5=van, 6=truck, 9=bus
    # 简化为: 0=vehicle
    vehicle_cats = {4, 5, 6, 9}

    count = 0
    for img_file in sorted(glob(os.path.join(src_images, '*.jpg'))):
        fname = os.path.basename(img_file)
        ann_file = os.path.join(src_anns, fname.replace('.jpg', '.txt'))
        if not os.path.exists(ann_file):
            continue

        img = cv2.imread(img_file)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 解析VisDrone标注
        labels = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                x, y, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cat = int(parts[5])

                if cat not in vehicle_cats:
                    continue
                if bw < 10 or bh < 10:
                    continue

                # 转YOLO格式(归一化中心点+宽高)
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                # 裁剪到[0,1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = min(nw, 1)
                nh = min(nh, 1)

                labels.append(f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')

        if not labels:
            continue

        # 复制图片和标签
        shutil.copy2(img_file, os.path.join(img_out, fname))
        with open(os.path.join(lbl_out, fname.replace('.jpg', '.txt')), 'w') as f:
            f.write('\n'.join(labels) + '\n')
        count += 1

    return count


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    visdrone_val = os.path.expanduser('~/datasets/VisDrone/VisDrone2019-DET-val')
    dataset_dir = os.path.join(base, 'visdrone_dataset')

    if not os.path.exists(os.path.join(visdrone_val, 'images')):
        print('VisDrone验证集不存在，请先下载')
        return

    # 转换标注格式
    print('转换VisDrone标注到YOLO格式...')

    # 用前400张做训练，后148张做验证
    images_dir = os.path.join(visdrone_val, 'images')
    anns_dir = os.path.join(visdrone_val, 'annotations')

    all_images = sorted(glob(os.path.join(images_dir, '*.jpg')))
    train_images = all_images[:400]
    val_images = all_images[400:]

    # 创建临时分割目录
    tmp_train_img = os.path.join(dataset_dir, 'tmp_train_img')
    tmp_train_ann = os.path.join(dataset_dir, 'tmp_train_ann')
    tmp_val_img = os.path.join(dataset_dir, 'tmp_val_img')
    tmp_val_ann = os.path.join(dataset_dir, 'tmp_val_ann')

    for d in [tmp_train_img, tmp_train_ann, tmp_val_img, tmp_val_ann]:
        os.makedirs(d, exist_ok=True)

    for f in train_images:
        fname = os.path.basename(f)
        shutil.copy2(f, os.path.join(tmp_train_img, fname))
        ann = os.path.join(anns_dir, fname.replace('.jpg', '.txt'))
        if os.path.exists(ann):
            shutil.copy2(ann, os.path.join(tmp_train_ann, fname.replace('.jpg', '.txt')))

    for f in val_images:
        fname = os.path.basename(f)
        shutil.copy2(f, os.path.join(tmp_val_img, fname))
        ann = os.path.join(anns_dir, fname.replace('.jpg', '.txt'))
        if os.path.exists(ann):
            shutil.copy2(ann, os.path.join(tmp_val_ann, fname.replace('.jpg', '.txt')))

    n_train = convert_visdrone_to_yolo(tmp_train_img, tmp_train_ann, dataset_dir, 'train')
    n_val = convert_visdrone_to_yolo(tmp_val_img, tmp_val_ann, dataset_dir, 'val')

    print(f'训练: {n_train}, 验证: {n_val}')

    # 清理临时目录
    for d in [tmp_train_img, tmp_train_ann, tmp_val_img, tmp_val_ann]:
        shutil.rmtree(d, ignore_errors=True)

    # 写dataset.yaml
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f'path: {dataset_dir}\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('nc: 1\n')
        f.write("names: ['vehicle']\n")

    # 训练
    print('\n开始训练 YOLOv8n (VisDrone车辆检测)...')
    model = YOLO('yolov8n.pt')
    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=8,
        device='cpu',
        name='visdrone_vehicle',
        patience=10,
        degrees=15,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        workers=0,
    )

    # 保存
    best_pt = model.trainer.best
    dst = os.path.join(base, 'best_visdrone.pt')
    shutil.copy2(str(best_pt), dst)
    print(f'\n训练完成! 模型: {dst}')

    # ONNX导出
    best_model = YOLO(best_pt)
    best_model.export(format='onnx', imgsz=640, simplify=True)
    onnx_src = str(best_pt).replace('.pt', '.onnx')
    if os.path.exists(onnx_src):
        shutil.copy2(onnx_src, dst.replace('.pt', '.onnx'))
        print(f'ONNX导出: {dst.replace(".pt", ".onnx")}')


if __name__ == '__main__':
    main()
