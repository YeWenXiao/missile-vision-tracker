"""
从RTSP实时截图补充训练数据

用法: python capture_more.py

操作:
  空格 = 截图保存
  q = 退出

拍摄建议（每种姿态拍5-10张）:
  - 盒子竖放、横放、斜放
  - 盒子远/中/近距离
  - 盒子在画面左/中/右
  - 不同背景（桌上、手持、地上）
  - 部分遮挡
"""

import cv2
import os
import time

rtsp_url = 'rtsp://192.168.144.25:8554/main.264'
images_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'images')
os.makedirs(images_dir, exist_ok=True)

# 统计已有图片数
existing = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
count = existing

print('连接RTSP...')
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print('无法连接RTSP!')
    exit()

print(f'已连接! 已有 {existing} 张图片')
print('空格=截图 q=退出')
print('请拿盒子各种角度摆放拍摄!\n')

cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    vis = frame.copy()
    cv2.putText(vis, f'Saved: {count - existing} | Total: {count} | SPACE=capture Q=quit',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Capture', vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        fname = f'{count:04d}.jpg'
        cv2.imwrite(os.path.join(images_dir, fname), frame)
        print(f'  已保存: {fname}')
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

new_count = count - existing
print(f'\n新增 {new_count} 张图片')
if new_count > 0:
    print(f'下一步: python label_tool.py  (标注新图片)')
    print(f'然后:   python train.py       (重新训练)')
