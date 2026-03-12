# V2.0 可用公开数据集

> 调研时间: 2026-03-12

## 优先级排序

| 优先级 | 数据集 | 用途 | 规模 |
|--------|--------|------|------|
| 1 | **VRAI** | 无人机航拍车辆重识别（同一目标多视角） | 137K图/13K身份 |
| 2 | **VRU** | 最大开源UAV车辆ReID | 172K图/15K身份 |
| 3 | **VisDrone** | 通用无人机目标检测训练 | 10K图+260K帧 |
| 4 | **UAV123** | 航拍单目标追踪基准 | 123序列/110K帧 |
| 5 | **UAVDT** | 多高度车辆检测 | 80K帧 |
| 6 | **DOTA v2.0** | 航拍小目标检测 | 11K图 |

## 详细信息

### 1. VRAI — 最推荐，直接匹配V2.0需求
- GitHub: https://github.com/JiaoBL1234/VRAI-Dataset
- 论文: ICCV 2019
- 内容: 同一车辆从不同无人机角度拍摄的多张图片，带身份标签
- **核心价值**: 直接用于验证颜色直方图+Hu矩+模板匹配的特征提取管线
- 免费(学术)

### 2. VRU (Vehicle ReID based on UAV)
- GitHub: https://github.com/GeoX-Lab/ReID
- 内容: 5架无人机拍摄，多场景多时段多天气
- **核心价值**: 最大规模开源UAV车辆ReID，多尺度多视角
- 完全开源

### 3. VisDrone
- GitHub: https://github.com/VisDrone/VisDrone-Dataset
- Kaggle: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset
- 内容: 14个城市，10类目标，260万+标注框
- **核心价值**: YOLO检测器训练的最佳航拍数据集
- CC BY-NC-SA

### 4. UAV123
- 官网: https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx
- 内容: 123个航拍追踪视频序列
- **核心价值**: 追踪模块的基准测试

### 5. UAVDT
- 官网: https://sites.google.com/view/grli-uavdt
- 内容: 80K帧，带高度/遮挡/天气属性标注
- **核心价值**: 多高度检测，匹配远/中/近需求

### 6. DOTA v2.0
- 官网: https://captain-whu.github.io/DOTA/
- 内容: 11K航拍图，18类，旋转框标注
- **核心价值**: 小目标+旋转不变检测

## 其他备选

- **DroneVehicle**: RGB+红外配对，28K图 (https://github.com/VisDrone/DroneVehicle)
- **Stanford Drone**: 斯坦福校园俯视追踪，20K目标 (https://cvgl.stanford.edu/projects/uav_data/)
- **PRAI-1581**: 无人机行人ReID，39K图/1.6K身份 (https://github.com/stormyoung/PRAI-1581)

## 本地训练方案

V2.0的YOLO检测器可以用VisDrone预训练，特征匹配管线用VRAI/VRU验证。
不需要接入摄像头到开发机——用这些数据集就够了。
