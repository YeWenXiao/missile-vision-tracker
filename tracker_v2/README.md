# 弹载视觉追踪系统 V2.0

## 核心升级

V2.0基于照片集特征匹配，实现"接收目标照片→搜索→锁定→末端追踪"全流程。

| 特性 | V1.5 | V2.0 |
|------|------|------|
| 目标识别 | 仅YOLO类别 | YOLO + HSV颜色 + Hu矩 + ORB纹理 |
| 状态机 | 2状态(扫描/追踪) | 3状态(搜索/追踪/末端) |
| Zoom策略 | 目标小→放大 | 接近式自动递减(6→4→3→2→1) |
| PID控制 | 固定参数 | 自适应(远/中/近/末端) |
| 代码结构 | 单文件1133行 | 10个模块 |
| 特征验证 | 无 | 持续验证防跟错 |
| 测试 | 无 | 离线测试+鲁棒性测试+基准测试 |

## 文件结构

```
tracker_v2/
├── main.py              # 入口 + 三状态机主循环
├── config.py            # 所有参数集中管理
├── gimbal.py            # SIYI A8mini云台控制
├── pid.py               # 自适应PID控制器
├── video.py             # RTSP视频流读取
├── detector.py          # YOLO目标检测器
├── feature_bank.py      # 照片集特征提取(HSV+Hu矩+多尺度模板)
├── target_matcher.py    # 多特征融合匹配(颜色+形状+ORB+YOLO)
├── zoom_manager.py      # 接近式zoom自动递减
├── web.py               # Web MJPEG监控
├── target_photos/       # 目标照片集
├── best.pt              # YOLO模型权重
├── deploy_jetson.sh     # Jetson部署脚本
├── benchmark.py         # 性能基准测试
├── test_matching.py     # 离线特征匹配测试
├── test_robustness.py   # 鲁棒性测试
└── train_visdrone.py    # VisDrone数据集训练
```

## 快速开始

### 1. 准备目标照片集

```
target_photos/
├── far_01.jpg           # 远距离目标照片
├── mid_01.jpg           # 中距离
├── near_01.jpg          # 近距离
└── target_info.json     # 标注信息 {"crops": {"far_01.jpg": [x1,y1,x2,y2]}}
```

### 2. 运行

```bash
# 实机运行(连接A8mini云台)
python main.py --photos target_photos/

# 视频文件测试
python main.py --source video.mp4 --no_gimbal --photos target_photos/

# 无照片集(仅YOLO)
python main.py --no_gimbal
```

### 3. 测试

```bash
# 离线特征匹配测试
python test_matching.py --photos target_photos/ --frames test_frames/

# 鲁棒性测试(尺度/旋转/亮度/噪声/模糊/遮挡)
python test_robustness.py

# 性能基准
python benchmark.py
```

## 三状态机

```
SEARCH ──找到目标──→ TRACK ──目标>50%──→ TERMINAL
  ↑                    │                    │
  └──丢失超时(5s)──────┘    目标缩小(<35%)──┘
```

| 状态 | 行为 | 触发 |
|------|------|------|
| SEARCH | 云台扫描 + YOLO + 特征匹配 | 启动 / 丢失超时 |
| TRACK | PID居中 + zoom递减 + 持续验证 | 找到匹配目标 |
| TERMINAL | 最高精度PID + zoom=1x | 目标占画面>50% |

## 特征匹配管线

每个YOLO检测框经过4维特征评分：

| 特征 | 权重 | 耗时 | 用途 |
|------|------|------|------|
| HSV颜色直方图 | 30% | ~0.2ms | 颜色匹配 |
| Hu矩 | 20% | ~0.1ms | 形状匹配(旋转不变) |
| ORB纹理 | 30% | ~2.3ms | 纹理区分(防假阳性) |
| YOLO置信度 | 20% | 0ms | 检测可信度 |

综合评分>0.55确认为目标，追踪中每秒验证一次(>0.35)防跟错。

## 性能

本机CPU(开发机):
- 完整管线: ~26ms/帧 = 38 FPS
- YOLO: 15ms, 特征匹配: 3ms, 模板搜索: 63ms

Jetson Orin Nano(TensorRT预估):
- 完整管线: ~23ms/帧 = **44 FPS**
- YOLO(TensorRT FP16): 20ms, 特征匹配: 3ms

## Jetson部署

```bash
# 开发机上运行
bash deploy_jetson.sh

# Jetson上导出TensorRT
ssh nvidia@192.168.5.28
cd ~/tracker_v2
yolo export model=best.pt format=engine imgsz=448 device=0
# 修改config.py: MODEL_PATH = 'best.engine'
python3 main.py
```

## 参数调优

所有参数在 `config.py` 中集中管理：

- `MATCH_THRESHOLD`: 搜索阶段确认阈值(当前0.55)
- `VERIFY_THRESHOLD`: 追踪验证阈值(当前0.35)
- `PID_PROFILES`: 各距离段PID参数
- `ZOOM_TABLE`: box_ratio→zoom级别映射
- `SEARCH_ZOOM_SCHEDULE`: 搜索阶段zoom时间表
