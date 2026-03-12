"""
最简单的YOLOv5目标检测测试

用法:
  python test_detect.py               # 用ref_6x.jpg测试
  python test_detect.py image.jpg     # 指定图片
  python test_detect.py rtsp://xxx    # RTSP视频流
"""

import sys
import os
import shutil

# 复制模型到当前目录
model_src = r'C:\Users\MS\Desktop\3588\runs\detect\crtk_box\weights\best.pt'
model_dst = os.path.join(os.path.dirname(__file__), 'best.pt')
if not os.path.exists(model_dst) and os.path.exists(model_src):
    shutil.copy2(model_src, model_dst)
    print(f'已复制模型: best.pt')

from ultralytics import YOLO

# 加载模型
model = YOLO(model_dst)
print(f'模型已加载: best.pt')

# 输入源
if len(sys.argv) > 1:
    source = sys.argv[1]
else:
    # 默认用ref_6x.jpg测试
    source = r'C:\Users\MS\Desktop\3588\ref_6x.jpg'

print(f'输入: {source}')

# 推理
results = model(source, conf=0.3, show=True, save=True)

# 打印结果
for r in results:
    if r.boxes and len(r.boxes) > 0:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            print(f'检测到目标! 置信度: {conf:.3f} 位置: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')
    else:
        print('未检测到目标')

print('\n结果图片保存在 runs/detect/ 目录下')
print('按任意键关闭窗口')
