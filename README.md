# 弹载计算机视觉追踪系统

基于 TW-T201S (RK3588) 机载计算机与 SIYI A8mini 云台相机的视觉目标追踪系统。

## 功能

- YOLOv5 目标检测（支持 ONNX / RKNN）
- PID 云台控制，保持目标画面居中
- 智能 Zoom 管理（自动放大/缩回）
- 运动预测 + 丢失恢复（沿消失方向追踪）
- Web MJPEG 实时监控
- 自动扫描搜索

## 硬件

| 硬件 | 型号 | IP |
|------|------|----|
| 机载计算机 | TW-T201S (RK3588) | 192.168.5.28 |
| 云台相机 | SIYI A8mini | 192.168.144.25 |

## 快速开始

```bash
# 安装依赖
pip install ultralytics opencv-python numpy

# 运行追踪（连接A8mini）
python tracker.py

# 无云台模式（纯检测测试）
python tracker.py --no_gimbal

# 调整参数
python tracker.py --conf 0.4 --zoom_level 4
```

运行后浏览器打开 `http://127.0.0.1:8080` 查看实时画面。

## 文件说明

| 文件 | 说明 |
|------|------|
| `tracker.py` | 主程序：YOLO检测 + 云台追踪 + Zoom |
| `demo.py` | 简易Demo：YOLO检测 + 显示 |
| `train.py` | YOLOv5 模型训练脚本 |
| `label_tool.py` | 数据标注工具 |
| `capture_more.py` | 数据采集工具 |
| `test_detect.py` | 检测测试脚本 |
| `best.pt` / `best.onnx` | 训练好的模型 |
| `dataset/` | 训练数据集（图片+标注） |
| `runs/` | 训练/推理结果 |

## 技术文档

详见 `弹载计算机视觉追踪_技术文档_20260312.docx`
