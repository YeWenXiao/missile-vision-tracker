#!/bin/bash
# V2.0 部署到 Jetson Orin Nano
#
# 在开发机(Windows)上运行:
#   bash deploy_jetson.sh
#
# 前置条件:
#   - Jetson SSH: nvidia@192.168.5.28
#   - Jetson已安装: python3, opencv, ultralytics, numpy

JETSON_USER="nvidia"
JETSON_IP="192.168.5.28"
JETSON_DIR="~/tracker_v2"

echo "========================================="
echo "  V2.0 部署到 Jetson Orin Nano"
echo "========================================="

# 创建远程目录
echo "[1/4] 创建远程目录..."
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ${JETSON_DIR}/target_photos"

# 传输代码
echo "[2/4] 传输代码..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
scp "${SCRIPT_DIR}/main.py" \
    "${SCRIPT_DIR}/config.py" \
    "${SCRIPT_DIR}/gimbal.py" \
    "${SCRIPT_DIR}/pid.py" \
    "${SCRIPT_DIR}/video.py" \
    "${SCRIPT_DIR}/detector.py" \
    "${SCRIPT_DIR}/feature_bank.py" \
    "${SCRIPT_DIR}/target_matcher.py" \
    "${SCRIPT_DIR}/zoom_manager.py" \
    "${SCRIPT_DIR}/web.py" \
    ${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/

# 传输模型
echo "[3/4] 传输模型..."
scp "${SCRIPT_DIR}/best.pt" ${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/ 2>/dev/null
scp "${SCRIPT_DIR}/best.onnx" ${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/ 2>/dev/null

# 传输照片集
echo "[4/4] 传输照片集..."
scp "${SCRIPT_DIR}/target_photos/"* ${JETSON_USER}@${JETSON_IP}:${JETSON_DIR}/target_photos/ 2>/dev/null

echo ""
echo "========================================="
echo "  部署完成!"
echo "========================================="
echo ""
echo "SSH到Jetson运行:"
echo "  ssh ${JETSON_USER}@${JETSON_IP}"
echo "  cd ${JETSON_DIR}"
echo "  python3 main.py"
echo ""
echo "TensorRT导出(在Jetson上执行):"
echo "  yolo export model=best.pt format=engine imgsz=448 device=0"
echo "  # 然后修改config.py MODEL_PATH='best.engine'"
echo ""
echo "Web监控: http://${JETSON_IP}:8080/"
