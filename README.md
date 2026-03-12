# 弹载计算机视觉追踪系统

基于 TW-T201S (RK3588) 机载计算机与 SIYI A8mini 云台相机的视觉目标追踪系统。

## 版本

**v1.0** — 基础追踪功能已实现，已知问题：目标易丢失、摄像头响应较慢

## 功能

- YOLOv5 实时目标检测（异步推理，不阻塞主循环）
- PID 云台控制，保持目标画面居中
- 智能 Zoom 管理（目标居中后自动放大，丢失时自动缩回）
- 运动预测 + 丢失恢复（记录消失方向，沿该方向追踪搜索）
- 接近速度控制（目标越近/越大，云台速度越慢越精确）
- Web MJPEG 实时监控（浏览器查看画面 + 云台控制按钮）
- 自动扫描搜索（多俯仰角巡视）

## 硬件

| 硬件 | 型号 | IP |
|------|------|----|
| 机载计算机 | TW-T201S (RK3588) | 192.168.5.28 |
| 云台相机 | SIYI A8mini | 192.168.144.25 |
| 视频流 | RTSP H.264 1280x720 | rtsp://192.168.144.25:8554/main.264 |

## 快速开始

```bash
# 安装依赖
pip install ultralytics opencv-python numpy

# 运行追踪（连接A8mini云台）
python tracker.py

# 无云台模式（纯检测测试）
python tracker.py --no_gimbal

# 不自动zoom
python tracker.py --no_zoom

# 调整置信度和zoom级别
python tracker.py --conf 0.4 --zoom_level 4
```

运行后浏览器打开 `http://127.0.0.1:8080` 查看实时画面。

## 文件说明

| 文件 | 说明 |
|------|------|
| `tracker.py` | 主程序：YOLO异步检测 + PID云台追踪 + 智能Zoom |
| `demo.py` | 简易Demo：YOLO检测 + 显示 |
| `train.py` | YOLOv5 模型训练脚本（自动下载预训练底模） |
| `label_tool.py` | 数据标注工具 |
| `capture_more.py` | 数据采集工具 |
| `best.pt` / `best.onnx` | 训练好的目标检测模型 |
| `dataset/` | 训练数据集（196张图片 + YOLO格式标注） |
| `runs/detect/target_v27/` | 最终训练结果（权重、曲线图、混淆矩阵） |

## 重新训练模型

```bash
# 1. 采集更多数据
python capture_more.py

# 2. 标注数据
python label_tool.py

# 3. 训练（自动下载yolov5nu.pt底模）
python train.py
```

## 技术文档

详见 `弹载计算机视觉追踪.docx`

## 已知问题（v1.0）

- 目标快速移动时容易丢失
- 摄像头云台响应存在延迟
- PID参数仍需针对实际场景进一步调优
