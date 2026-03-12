"""
从测试帧生成测试视频，用于main.py端到端测试
"""

import os
import cv2
from glob import glob

frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_frames')
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_video.avi')

files = sorted(glob(os.path.join(frames_dir, 'frame_*.jpg')))
if not files:
    print('No test frames found')
    exit(1)

first = cv2.imread(files[0])
h, w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(out_path, fourcc, 10.0, (w, h))

# 每帧重复5次，制造更长的视频(约10秒)
for f in files:
    img = cv2.imread(f)
    if img is not None:
        for _ in range(5):
            writer.write(img)

writer.release()
print(f'测试视频已生成: {out_path} ({len(files)*5} frames, {w}x{h})')
