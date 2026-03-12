"""
完整追踪系统: YOLO检测 + 云台追踪 + Zoom放大 + Web监控

工作流程:
  1. 连接A8mini RTSP视频流
  2. YOLO实时检测目标
  3. 检测到目标 → 云台转向目标 → zoom放大 → 持续追踪保持居中
  4. 目标丢失 → zoom缩回 → 继续扫描

用法:
  python tracker.py                    # 默认连接A8mini
  python tracker.py --no_gimbal        # 无云台模式(纯检测)
  python tracker.py --conf 0.4         # 调低置信度
  python tracker.py --no_zoom          # 不自动zoom
"""

import os
import sys
import time
import json
import argparse
import socket
import struct
import threading
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from ultralytics import YOLO


# ============================================================================
# SIYI A8mini 云台控制
# ============================================================================

class SIYIGimbal:
    """SIYI A8mini 云台控制器（SDK V0.1.1 协议）"""

    CRC16_TAB = [
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
        0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
        0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
        0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
        0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
        0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
        0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
        0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
        0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
        0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
        0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
        0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
        0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
        0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
        0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
        0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
        0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
        0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
        0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
        0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
        0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
        0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
        0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
        0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
        0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
        0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
        0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
        0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
        0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
        0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
        0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
        0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0,
    ]

    def __init__(self, ip='192.168.144.25', port=37260):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.seq = 0

    @classmethod
    def _crc16(cls, data):
        crc = 0
        for b in data:
            temp = (crc >> 8) & 0xFF
            crc = (crc << 8) ^ cls.CRC16_TAB[b ^ temp]
            crc &= 0xFFFF
        return crc

    def _send(self, cmd_id, data=b''):
        stx = bytes([0x55, 0x66])
        ctrl = bytes([0x01])
        data_len = struct.pack('<H', len(data))
        seq = struct.pack('<H', self.seq)
        cmd = bytes([cmd_id])
        packet_body = stx + ctrl + data_len + seq + cmd + data
        crc = struct.pack('<H', self._crc16(packet_body))
        packet = packet_body + crc
        self.seq = (self.seq + 1) % 65536
        try:
            self.sock.sendto(packet, (self.ip, self.port))
        except Exception as e:
            print(f'[云台] 发送失败: {e}')

    def set_speed(self, yaw, pitch):
        yaw = max(-100, min(100, int(yaw)))
        pitch = max(-100, min(100, int(pitch)))
        self._send(0x07, bytes([yaw & 0xFF, pitch & 0xFF]))

    def stop(self):
        self.set_speed(0, 0)

    def center(self):
        self._send(0x08, bytes([0x01]))

    def set_angle(self, yaw_deg, pitch_deg):
        data = struct.pack('<hh', int(yaw_deg * 10), int(pitch_deg * 10))
        self._send(0x0E, data)

    def zoom_in(self):
        self._send(0x05, bytes([0x01]))

    def zoom_out(self):
        self._send(0x05, bytes([0xFF]))

    def zoom_stop(self):
        self._send(0x05, bytes([0x00]))

    def close(self):
        self.stop()
        self.zoom_stop()
        self.sock.close()


# ============================================================================
# PID 控制器
# ============================================================================

class PID:
    def __init__(self, kp=0.8, ki=0.01, kd=0.1, max_out=80):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_err = 0.0

    def compute(self, error, dt=0.033):
        self.integral += error * dt
        self.integral = max(-50, min(50, self.integral))
        deriv = (error - self.prev_err) / dt if dt > 0 else 0
        self.prev_err = error
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        return max(-self.max_out, min(self.max_out, out))

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0


# ============================================================================
# YOLO 异步检测器（后台线程推理，主循环不阻塞）
# ============================================================================

class AsyncDetector:
    """后台线程运行YOLO推理，主循环只取结果不等待"""
    def __init__(self, model, conf=0.4, imgsz=448):
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.lock = threading.Lock()
        self.input_frame = None
        self.input_ready = threading.Event()
        self.result = None       # 最新检测结果 (boxes, confs) 或 None
        self.result_fresh = False # 是否有未读取的新结果
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def submit(self, frame):
        """提交一帧给后台检测（非阻塞）"""
        with self.lock:
            self.input_frame = frame.copy()
        self.input_ready.set()

    def get_result(self):
        """获取最新结果，如果有新的返回 (boxes_xyxy, confs)，否则返回 None"""
        with self.lock:
            if self.result_fresh:
                self.result_fresh = False
                return self.result
        return None

    def _loop(self):
        while self.running:
            self.input_ready.wait(timeout=1.0)
            self.input_ready.clear()
            with self.lock:
                frame = self.input_frame
            if frame is None:
                continue
            results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
            det = None
            if results and results[0].boxes and len(results[0].boxes) > 0:
                det = (results[0].boxes.xyxy.cpu().numpy(),
                       results[0].boxes.conf.cpu().numpy())
            with self.lock:
                self.result = det
                self.result_fresh = True

    def stop(self):
        self.running = False


# ============================================================================
# RTSP 视频流（后台线程读取，总是拿最新帧）
# ============================================================================

class RTSPReader:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        for _ in range(50):
            time.sleep(0.1)
            if self.frame is not None:
                return True
        return False

    def _loop(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f'[RTSP] 无法连接: {self.url}')
            self.running = False
            return
        print(f'[RTSP] 已连接')
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)
        cap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


# ============================================================================
# Web MJPEG 服务器
# ============================================================================

HTML_PAGE = '''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>A8mini Tracker</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#111; color:#0f0; font-family:monospace; display:flex; flex-direction:column; align-items:center; }
h1 { font-size:16px; padding:8px; }
#container { position:relative; display:inline-block; }
#video { border:2px solid #0f0; max-width:95vw; max-height:85vh; }
#status { position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.7); color:#0f0; padding:6px 12px; font-size:14px; border-radius:4px; }
#btns { padding:5px; }
#btns button { background:#333; color:#0f0; border:1px solid #0f0; padding:6px 16px; margin:0 5px; cursor:pointer; font-family:monospace; }
#btns button:hover { background:#0f0; color:#111; }
</style>
</head><body>
<h1>SIYI A8mini Auto Tracker</h1>
<div id="btns">
    <button onclick="doAction('center')">云台回中</button>
    <button onclick="doAction('rescan')">重新扫描</button>
    <button onclick="doAction('zoom_in')">Zoom+</button>
    <button onclick="doAction('zoom_out')">Zoom-</button>
</div>
<div id="container">
    <img id="video" src="/stream">
    <div id="status">启动中...</div>
</div>
<script>
function doAction(act) {
    fetch('/' + act).then(r => r.json()).then(d => {
        document.getElementById('status').textContent = d.msg || 'OK';
    });
}
setInterval(function() {
    fetch('/status').then(r => r.json()).then(d => {
        document.getElementById('status').textContent = d.text;
    }).catch(() => {});
}, 500);
</script>
</body></html>'''


class WebServer:
    def __init__(self, port=8080):
        self.port = port
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.status_text = '启动中...'
        self.callbacks = {}

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split('?')[0]

                if path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(HTML_PAGE.encode('utf-8'))

                elif path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.end_headers()
                    try:
                        while server_ref.running:
                            with server_ref.lock:
                                f = server_ref.frame
                            if f is not None:
                                small = cv2.resize(f, (640, 360))
                                ret, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                                if ret:
                                    data = jpg.tobytes()
                                    self.wfile.write(b'--frame\r\n')
                                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                    self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                                    self.wfile.write(b'\r\n')
                                    self.wfile.write(data)
                                    self.wfile.write(b'\r\n')
                            time.sleep(0.08)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                elif path == '/status':
                    self._json({'text': server_ref.status_text})

                elif path.lstrip('/') in server_ref.callbacks:
                    action = path.lstrip('/')
                    msg = server_ref.callbacks[action]()
                    self._json({'ok': True, 'msg': msg})

                else:
                    self.send_response(404)
                    self.end_headers()

            def _json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, format, *args):
                pass

        self.server = HTTPServer(('0.0.0.0', port), Handler)

    def start(self):
        self.running = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def stop(self):
        self.running = False
        self.server.shutdown()


# ============================================================================
# 自动扫描
# ============================================================================

class Scanner:
    def __init__(self, gimbal):
        self.gimbal = gimbal
        self.active = True
        self.scan_speed = 12
        self.direction = 1
        self.step_time = 6.0
        self.last_switch = time.time()
        self.pitch_angles = [0, -20, -40]
        self.pitch_idx = 0
        self.sweep_count = 0

    def update(self):
        if not self.active or not self.gimbal:
            return
        now = time.time()
        if now - self.last_switch > self.step_time:
            self.direction *= -1
            self.sweep_count += 1
            self.last_switch = now

            if self.sweep_count % 2 == 0:
                self.pitch_idx = (self.pitch_idx + 1) % len(self.pitch_angles)
                pitch = self.pitch_angles[self.pitch_idx]
                self.gimbal.set_angle(0, pitch)
                print(f'[扫描] 俯仰: {pitch}°')

        self.gimbal.set_speed(self.scan_speed * self.direction, 0)

    def pause(self):
        self.active = False
        if self.gimbal:
            self.gimbal.stop()

    def resume(self):
        self.active = True
        self.last_switch = time.time()
        print('[扫描] 恢复扫描')


# ============================================================================
# 主系统
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='YOLO检测 + 云台追踪 + Zoom')
    parser.add_argument('--camera_ip', default='192.168.144.25')
    parser.add_argument('--rtsp_url', default=None)
    parser.add_argument('--model', default='best.pt')
    parser.add_argument('--conf', type=float, default=0.4, help='置信度阈值')
    parser.add_argument('--web', type=int, default=8080, help='Web端口')
    parser.add_argument('--no_gimbal', action='store_true', help='无云台模式')
    parser.add_argument('--no_scan', action='store_true', help='不自动扫描')
    parser.add_argument('--no_zoom', action='store_true', help='不自动zoom')
    parser.add_argument('--zoom_level', type=int, default=2,
                        help='追踪时zoom级别: 1=不zoom 2=2x 4=4x (默认2)')
    args = parser.parse_args()

    if args.rtsp_url is None:
        args.rtsp_url = f'rtsp://{args.camera_ip}:8554/main.264'

    # 模型路径
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)

    print('=' * 50)
    print('  SIYI A8mini 自动追踪系统')
    print('  YOLO检测 → 云台追踪 → Zoom放大')
    print('=' * 50)

    # 加载YOLO模型
    print(f'[模型] 加载: {model_path}')
    model = YOLO(model_path)
    print(f'[模型] 已加载, 置信度阈值: {args.conf}')

    # 异步检测器（YOLO推理在后台线程，不阻塞主循环）
    async_det = AsyncDetector(model, conf=args.conf, imgsz=448)
    async_det.start()

    # 云台
    gimbal = None
    if not args.no_gimbal:
        gimbal = SIYIGimbal(args.camera_ip)
        gimbal.center()
        time.sleep(1)
        gimbal.zoom_out()
        time.sleep(5)
        gimbal.zoom_stop()
        print(f'[云台] 已连接: {args.camera_ip}')
    else:
        print('[云台] 已禁用')

    # PID
    yaw_pid = PID(kp=18, ki=0.2, kd=1.5, max_out=50)
    pitch_pid = PID(kp=18, ki=0.2, kd=1.5, max_out=50)

    # 扫描器
    scanner = Scanner(gimbal) if not args.no_scan else None

    # RTSP
    print(f'[RTSP] 连接: {args.rtsp_url}')
    reader = RTSPReader(args.rtsp_url)
    if not reader.start():
        print('[RTSP] 连接失败!')
        sys.exit(1)
    print('[RTSP] 已连接')

    # Web
    web = WebServer(args.web)

    # 状态
    STATE_SCAN = 'SCANNING'
    STATE_TRACK = 'TRACKING'
    state = STATE_SCAN

    zoom_start_time = 0
    zoom_first_time = 0
    zoom_pause_time = 0
    zooming_in = False
    lost_count = 0
    track_count = 0
    last_det = None
    last_det_conf = 0.0

    # 非阻塞zoom计时器: zoom_out操作开始时间，0表示没有进行中的zoom_out
    zoom_out_start = 0
    zoom_out_duration = 0

    # === 运动预测 ===
    prev_center = None
    velocity = [0.0, 0.0]
    velocity_smooth = 0.75
    last_yaw_speed = 0.0
    last_pitch_speed = 0.0
    inertia_decay = 0.82
    approach_scale = 1.0
    lost_vel = [0.0, 0.0]         # 丢失时的速度方向
    lost_last_err = [0.0, 0.0]    # 丢失时目标偏离中心的方向

    fps = 0.0
    fps_count = 0
    fps_time = time.time()
    last_time = time.time()
    total_frames = 0
    last_submit_frame = 0  # 上次提交检测的帧号

    def do_center():
        nonlocal state, lost_count, track_count, last_det, zooming_in, zoom_out_start, zoom_out_duration
        state = STATE_SCAN
        lost_count = 0
        track_count = 0
        last_det = None
        zooming_in = False
        if gimbal:
            gimbal.stop()
            gimbal.center()
            gimbal.zoom_out()
            zoom_out_start = time.time()
            zoom_out_duration = 4.0
        if scanner:
            scanner.resume()
        yaw_pid.reset()
        pitch_pid.reset()
        return '云台已回中'

    def do_rescan():
        nonlocal state, lost_count, track_count, last_det, zooming_in, zoom_out_start, zoom_out_duration
        state = STATE_SCAN
        lost_count = 0
        track_count = 0
        last_det = None
        zooming_in = False
        if gimbal:
            gimbal.stop()
            gimbal.zoom_out()
            zoom_out_start = time.time()
            zoom_out_duration = 4.0
        if scanner:
            scanner.resume()
        yaw_pid.reset()
        pitch_pid.reset()
        return '重新扫描'

    def do_zoom_in():
        if gimbal:
            gimbal.zoom_in()
        return 'Zoom+'

    def do_zoom_out():
        if gimbal:
            gimbal.zoom_out()
        return 'Zoom-'

    web.callbacks = {
        'center': do_center,
        'rescan': do_rescan,
        'zoom_in': do_zoom_in,
        'zoom_out': do_zoom_out,
    }
    web.start()
    print(f'[Web] http://127.0.0.1:{args.web}/')

    # 录像
    save_path = os.path.join(os.path.dirname(__file__), 'tracker_output.avi')
    video_writer = None

    print(f'\n[系统] 运行中...')
    print(f'[系统] 录像保存到: {save_path}')
    print(f'[系统] 按 Ctrl+C 退出\n')

    try:
        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - last_time
            last_time = now
            total_frames += 1

            # FPS
            fps_count += 1
            if now - fps_time >= 1.0:
                fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now

            h, w = frame.shape[:2]
            det_box = None

            # === 非阻塞zoom_out计时器 ===
            if zoom_out_start > 0 and now - zoom_out_start >= zoom_out_duration:
                if gimbal:
                    gimbal.zoom_stop()
                zoom_out_start = 0
                zoom_out_duration = 0

            # ============== YOLO异步检测 ==============
            # 每帧都提交给后台线程（后台会自动用最新帧，旧的丢弃）
            if total_frames - last_submit_frame >= 2:
                async_det.submit(frame)
                last_submit_frame = total_frames

            # 非阻塞取结果
            det_result = async_det.get_result()
            if det_result is not None:
                boxes_xyxy, confs = det_result
                # 过滤太大的检测框
                valid_mask = []
                for i in range(len(confs)):
                    bx1, by1, bx2, by2 = boxes_xyxy[i]
                    bw = (bx2 - bx1) / w
                    bh = (by2 - by1) / h
                    valid_mask.append(max(bw, bh) < 0.40)

                if state == STATE_TRACK and last_det is not None:
                    lx1, ly1, lx2, ly2 = last_det
                    last_cx = (lx1 + lx2) / 2
                    last_cy = (ly1 + ly2) / 2
                    last_size = max(lx2 - lx1, ly2 - ly1)
                    max_dist = max(last_size * 3, 150)

                    best_idx = None
                    best_score = 0
                    for i in range(len(confs)):
                        if not valid_mask[i]:
                            continue
                        bx1, by1, bx2, by2 = boxes_xyxy[i]
                        bcx = (bx1 + bx2) / 2
                        bcy = (by1 + by2) / 2
                        dist = ((bcx - last_cx)**2 + (bcy - last_cy)**2)**0.5
                        if dist < max_dist:
                            score = float(confs[i])
                            if score > best_score:
                                best_score = score
                                best_idx = i

                    if best_idx is not None:
                        x1, y1, x2, y2 = boxes_xyxy[best_idx]
                        det_box = (int(x1), int(y1), int(x2), int(y2))
                        last_det = det_box
                        last_det_conf = float(confs[best_idx])
                else:
                    best_idx = None
                    best_score = 0
                    for i in range(len(confs)):
                        if not valid_mask[i]:
                            continue
                        score = float(confs[i])
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    if best_idx is None and len(confs) > 0:
                        best_idx = int(np.argmax(confs))
                    if best_idx is not None:
                        x1, y1, x2, y2 = boxes_xyxy[best_idx]
                        det_box = (int(x1), int(y1), int(x2), int(y2))
                        last_det = det_box
                        last_det_conf = float(confs[best_idx])

            # ============== 状态机 ==============

            if state == STATE_SCAN:
                if scanner:
                    scanner.update()

                if det_box is not None:
                    # 发现目标! 停止扫描，开始追踪
                    if scanner:
                        scanner.pause()
                    lost_count = 0
                    track_count = 0
                    prev_center = None
                    velocity[:] = [0.0, 0.0]
                    last_yaw_speed = 0.0
                    last_pitch_speed = 0.0
                    yaw_pid.reset()
                    pitch_pid.reset()
                    state = STATE_TRACK
                    print(f'[系统] 发现目标! conf:{last_det_conf:.2f} 切换到追踪模式')

            elif state == STATE_TRACK:
                # 如果本帧没检测到，用上一次的检测结果（容忍短暂丢失）
                active_box = det_box if det_box is not None else last_det

                if active_box is not None and (det_box is not None or lost_count < 15):
                    # ==================== 目标可见 → PID居中 ====================
                    x1, y1, x2, y2 = active_box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    err_x = (cx - w / 2) / (w / 2)  # -1 ~ +1
                    err_y = (cy - h / 2) / (h / 2)

                    if det_box is not None:
                        # === 刚从丢失状态找回目标 ===
                        was_lost = lost_count > 10
                        lost_count = 0
                        track_count += 1

                        # === 计算目标速度(归一化) ===
                        cx_norm = cx / w
                        cy_norm = cy / h
                        if prev_center is not None:
                            raw_vx = cx_norm - prev_center[0]
                            raw_vy = cy_norm - prev_center[1]
                            velocity[0] = velocity_smooth * velocity[0] + (1 - velocity_smooth) * raw_vx
                            velocity[1] = velocity_smooth * velocity[1] + (1 - velocity_smooth) * raw_vy
                        prev_center = (cx_norm, cy_norm)

                        # === 接近速度控制 ===
                        box_w_now = x2 - x1
                        box_h_now = y2 - y1
                        box_ratio_now = max(box_w_now / w, box_h_now / h)
                        approach_scale = max(0.3, min(1.0, 1.0 - box_ratio_now * 2.0))

                        # === PID + 运动预测 ===
                        yaw_speed = yaw_pid.compute(err_x, dt) if abs(err_x) > 0.03 else 0
                        pitch_speed = pitch_pid.compute(err_y, dt) if abs(err_y) > 0.03 else 0

                        yaw_speed *= approach_scale
                        pitch_speed *= approach_scale

                        predict_gain = 40 * approach_scale
                        yaw_speed += velocity[0] * predict_gain
                        pitch_speed += velocity[1] * predict_gain

                        # 边缘加速（平滑渐变）
                        edge_margin = 0.35
                        if abs(err_x) > edge_margin:
                            edge_extra = (abs(err_x) - edge_margin) / (1.0 - edge_margin)
                            yaw_speed += edge_extra * 12 * approach_scale * (1 if err_x > 0 else -1)
                        if abs(err_y) > edge_margin:
                            edge_extra = (abs(err_y) - edge_margin) / (1.0 - edge_margin)
                            pitch_speed += edge_extra * 12 * approach_scale * (1 if err_y > 0 else -1)

                        max_speed = 50 * approach_scale + 15
                        yaw_speed = max(-max_speed, min(max_speed, yaw_speed))
                        pitch_speed = max(-max_speed, min(max_speed, pitch_speed))

                        last_yaw_speed = yaw_speed
                        last_pitch_speed = pitch_speed

                        if gimbal:
                            gimbal.set_speed(yaw_speed, -pitch_speed)

                        # === 找回后的zoom策略 ===
                        if was_lost and gimbal and not args.no_zoom:
                            # 刚找回目标，先确保居中再考虑zoom
                            zoom_pause_time = now  # 冷却：找回后至少等2秒再zoom
                            if zooming_in:
                                gimbal.zoom_stop()
                                zooming_in = False
                            print(f'[追踪] 目标找回! 先居中稳定')

                    else:
                        # 本帧没有新检测 → 用记忆PID控制
                        lost_count += 1
                        if lost_count < 10:
                            yaw_speed = yaw_pid.compute(err_x, dt) if abs(err_x) > 0.03 else 0
                            pitch_speed = pitch_pid.compute(err_y, dt) if abs(err_y) > 0.03 else 0
                            predict_gain = 40 * approach_scale
                            yaw_speed += velocity[0] * predict_gain
                            pitch_speed += velocity[1] * predict_gain
                            yaw_speed = max(-50, min(50, yaw_speed))
                            pitch_speed = max(-50, min(50, pitch_speed))
                            last_yaw_speed = yaw_speed
                            last_pitch_speed = pitch_speed
                            if gimbal:
                                gimbal.set_speed(yaw_speed, -pitch_speed)
                        else:
                            last_yaw_speed *= inertia_decay
                            last_pitch_speed *= inertia_decay
                            if gimbal:
                                if abs(last_yaw_speed) > 2 or abs(last_pitch_speed) > 2:
                                    gimbal.set_speed(last_yaw_speed, -last_pitch_speed)
                                else:
                                    gimbal.set_speed(0, 0)

                    # ==================== 智能zoom管理 ====================
                    if not args.no_zoom and gimbal and det_box is not None:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_ratio = max(box_w / w, box_h / h)
                        vel_magnitude = (velocity[0]**2 + velocity[1]**2)**0.5

                        if zooming_in:
                            # 正在zoom — 每0.5秒暂停一次让PID居中
                            if now - zoom_start_time > 0.5:
                                gimbal.zoom_stop()
                                zooming_in = False
                                zoom_pause_time = now
                            # 最大zoom 6秒
                            if now - zoom_first_time > 6.0:
                                gimbal.zoom_stop()
                                zooming_in = False
                                print(f'[Zoom] 达到最大zoom')
                            # 目标够大了就停
                            if box_ratio > 0.20:
                                gimbal.zoom_stop()
                                zooming_in = False
                                zoom_first_time = 0
                                print(f'[Zoom] 目标足够大({box_ratio:.0%})，停止zoom')
                        else:
                            # 没在zoom — 满足条件就zoom放大一点
                            center_ok = abs(err_x) < 0.10 and abs(err_y) < 0.10
                            size_small = box_ratio < 0.15  # 目标较小时zoom
                            cooldown_ok = now - zoom_pause_time > 2.0
                            stable = track_count > 15
                            not_fast = vel_magnitude < 0.010

                            if size_small and center_ok and cooldown_ok and stable and not_fast:
                                zooming_in = True
                                gimbal.zoom_in()
                                zoom_start_time = now
                                if zoom_first_time == 0:
                                    zoom_first_time = now
                                print(f'[Zoom] 目标居中({box_ratio:.0%})，zoom放大')

                    # 打印追踪信息（每秒一次）
                    if fps_count == 0:
                        marker = '' if det_box is not None else ' (记忆)'
                        box_w = x2 - x1
                        box_h = y2 - y1
                        box_pct = max(box_w / w, box_h / h)
                        zoom_str = ' ZOOM+' if zooming_in else ''
                        vel_mag = (velocity[0]**2 + velocity[1]**2)**0.5
                        a_scale = max(0.3, min(1.0, 1.0 - box_pct * 2.0))
                        print(f'[追踪] conf:{last_det_conf:.2f} '
                              f'err:({err_x:+.2f},{err_y:+.2f}) '
                              f'size:{box_pct:.0%} spd:{a_scale:.1f}x{zoom_str} '
                              f'vel:{vel_mag:.3f}{marker}')

                else:
                    # ==================== 目标丢失 ====================
                    lost_count += 1

                    # --- 记录消失方向（只记一次）---
                    if lost_count == 1:
                        # 保存消失时的速度方向和最后位置
                        lost_vel = [velocity[0], velocity[1]]
                        lost_last_err = [0.0, 0.0]
                        if last_det is not None:
                            lx1, ly1, lx2, ly2 = last_det
                            lost_last_err[0] = ((lx1+lx2)/2 - w/2) / (w/2)
                            lost_last_err[1] = ((ly1+ly2)/2 - h/2) / (h/2)

                    # --- 阶段0: 立即zoom缩回（非阻塞）---
                    if lost_count == 1 and gimbal and not args.no_zoom:
                        if zooming_in:
                            gimbal.zoom_stop()
                            zooming_in = False
                        if zoom_out_start == 0:
                            gimbal.zoom_out()
                            zoom_out_start = now
                            zoom_out_duration = 1.5
                        zoom_first_time = 0
                        print(f'[搜索] 丢失! zoom缩回, 沿消失方向追踪')

                    # --- 阶段1: 1~30帧 — 沿消失方向追踪 ---
                    if 1 <= lost_count <= 30 and gimbal:
                        # 用消失时的速度方向 + 最后偏移方向推算
                        chase_yaw = 0.0
                        chase_pitch = 0.0
                        # 速度方向权重更大
                        if abs(lost_vel[0]) > 0.003:
                            chase_yaw = 25 if lost_vel[0] > 0 else -25
                        elif abs(lost_last_err[0]) > 0.1:
                            chase_yaw = 18 if lost_last_err[0] > 0 else -18
                        if abs(lost_vel[1]) > 0.003:
                            chase_pitch = 18 if lost_vel[1] > 0 else -18
                        elif abs(lost_last_err[1]) > 0.1:
                            chase_pitch = 12 if lost_last_err[1] > 0 else -12
                        # 逐帧衰减追踪速度
                        decay = max(0.0, 1.0 - lost_count / 35.0)
                        gimbal.set_speed(chase_yaw * decay, -(chase_pitch * decay))

                    # --- 阶段2: 30~90帧 — 以消失方向为主轴左右摆动搜索 ---
                    elif 30 < lost_count <= 90 and gimbal:
                        sweep_period = 25
                        phase = (lost_count - 30) // sweep_period
                        # 主方向: 消失方向，副方向: 交替反转
                        if abs(lost_vel[0]) > 0.003:
                            base_dir = 1 if lost_vel[0] > 0 else -1
                        elif abs(lost_last_err[0]) > 0.1:
                            base_dir = 1 if lost_last_err[0] > 0 else -1
                        else:
                            base_dir = 1
                        sweep_dir = base_dir if phase % 2 == 0 else -base_dir
                        gimbal.set_speed(22 * sweep_dir, 0)
                        if lost_count == 31:
                            print(f'[搜索] 沿消失方向摆动搜索...')

                    # --- 阶段3: 90帧 — 再zoom缩回一次 ---
                    if lost_count == 90 and gimbal and not args.no_zoom:
                        print(f'[搜索] 扩大搜索范围，zoom继续缩回...')
                        gimbal.stop()
                        if zoom_out_start == 0:
                            gimbal.zoom_out()
                            zoom_out_start = now
                            zoom_out_duration = 2.0

                    # --- 阶段3续: 90~180帧 — 更大范围摆动 ---
                    if 90 < lost_count <= 180 and gimbal:
                        sweep_period = 30
                        sweep_dir = 1 if ((lost_count - 90) // sweep_period) % 2 == 0 else -1
                        gimbal.set_speed(28 * sweep_dir, 0)

                    # --- 阶段4: 180帧 — 放弃，就地扫描 ---
                    if lost_count > 180:
                        print(f'[系统] 目标丢失({lost_count}帧)，就地扫描')
                        state = STATE_SCAN
                        track_count = 0
                        last_det = None
                        prev_center = None
                        velocity[:] = [0.0, 0.0]
                        last_yaw_speed = 0.0
                        last_pitch_speed = 0.0
                        zooming_in = False
                        zoom_first_time = 0
                        if gimbal:
                            gimbal.stop()
                            if zoom_out_start == 0:
                                gimbal.zoom_out()
                                zoom_out_start = now
                                zoom_out_duration = 3.0
                        if scanner:
                            scanner.resume()
                        yaw_pid.reset()
                        pitch_pid.reset()

            # ============== OSD绘制 ==============
            vis = frame.copy()

            # 十字架（画面中心）
            cs = 25
            cv2.line(vis, (w//2 - cs, h//2), (w//2 + cs, h//2), (0, 0, 255), 2)
            cv2.line(vis, (w//2, h//2 - cs), (w//2, h//2 + cs), (0, 0, 255), 2)

            # 画目标框 — 优先用当前检测，否则用上一次记忆
            show_box = det_box if det_box is not None else (last_det if state == STATE_TRACK else None)
            if show_box is not None:
                x1, y1, x2, y2 = show_box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # 当前检测=绿色实框，记忆=黄色虚线
                if det_box is not None:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f'{last_det_conf:.2f}', (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.line(vis, (w//2, h//2), (cx, cy), (0, 255, 255), 1)
                # 速度箭头（显示预测方向）
                if abs(velocity[0]) > 0.002 or abs(velocity[1]) > 0.002:
                    arrow_scale = 3000  # 放大速度向量用于显示
                    ax = int(cx + velocity[0] * arrow_scale)
                    ay = int(cy + velocity[1] * arrow_scale)
                    cv2.arrowedLine(vis, (cx, cy), (ax, ay), (255, 0, 255), 2, tipLength=0.3)
                status_color = (0, 255, 0) if det_box is not None else (0, 165, 255)
            elif state == STATE_TRACK and lost_count > 0:
                status_color = (0, 165, 255)
            else:
                status_color = (200, 200, 200)

            status_text = f'{state}'
            if zooming_in:
                status_text += ' ZOOM+'
            if state == STATE_TRACK and lost_count > 0:
                status_text += f' lost:{lost_count}'
            cv2.putText(vis, f'FPS:{fps:.0f} | {status_text}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            web.update_frame(vis)
            web.status_text = f'FPS:{fps:.0f} | {status_text}'

            # 录像写入
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(save_path, fourcc, 15.0, (w, h))
            video_writer.write(vis)

            # 本地显示（Windows有GUI）
            if total_frames == 1:
                cv2.namedWindow('Tracker', cv2.WINDOW_NORMAL)
            cv2.imshow('Tracker', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print('\n[系统] 用户退出')
    finally:
        if video_writer:
            video_writer.release()
            print(f'[录像] 已保存: {save_path}')
        async_det.stop()
        reader.stop()
        if gimbal:
            gimbal.stop()
            gimbal.zoom_stop()
            gimbal.close()
        web.stop()
        cv2.destroyAllWindows()
        print('[系统] 已退出')


if __name__ == '__main__':
    main()
