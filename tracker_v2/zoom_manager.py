"""
接近式Zoom管理器 — 载体持续接近目标，zoom自动递减
"""

import time
from config import (
    ZOOM_TABLE, ZOOM_PULSE_ON, ZOOM_PULSE_OFF, ZOOM_CHANGE_COOLDOWN,
)


class ZoomManager:
    """
    zoom只需单向递减（6→4→3→2→1），因为载体一直在接近。
    通过脉冲式zoom-out实现平滑过渡。
    """

    def __init__(self, gimbal):
        self.gimbal = gimbal
        self.current_zoom = 6       # 搜索阶段从最大zoom开始
        self.target_zoom = 6
        self.last_change_time = 0
        self.zooming = False        # True=正在zoom操作中
        self.zoom_start = 0
        self.enabled = True

    def update(self, box_ratio):
        """
        根据目标大小更新zoom
        每帧调用一次
        """
        if not self.enabled or not self.gimbal:
            return

        now = time.time()

        # 计算目标zoom级别
        new_target = 1
        for max_ratio, zoom_level in ZOOM_TABLE:
            if box_ratio < max_ratio:
                new_target = zoom_level
                break

        # zoom级别变化 → 开始zoom操作
        if new_target < self.target_zoom:
            if now - self.last_change_time > ZOOM_CHANGE_COOLDOWN:
                self.target_zoom = new_target
                print(f'[Zoom] 目标接近(size:{box_ratio:.1%})，zoom: {self.current_zoom}→{self.target_zoom}')

        # 需要zoom-out时，脉冲式执行
        if self.current_zoom > self.target_zoom:
            if not self.zooming:
                # 开始一次zoom-out脉冲
                self.gimbal.zoom_out()
                self.zooming = True
                self.zoom_start = now
            elif now - self.zoom_start > ZOOM_PULSE_ON:
                # 脉冲结束
                self.gimbal.zoom_stop()
                self.zooming = False
                self.current_zoom = max(self.current_zoom - 1, self.target_zoom)
                self.last_change_time = now

        elif self.zooming:
            # 已到达目标zoom，确保停止
            self.gimbal.zoom_stop()
            self.zooming = False

    def set_search_zoom(self, elapsed_time, schedule):
        """
        搜索阶段根据时间表设置zoom
        """
        if not self.gimbal:
            return

        target = 1
        for time_limit, zoom_level in schedule:
            if elapsed_time < time_limit:
                target = zoom_level
                break

        if target != self.current_zoom:
            if target < self.current_zoom:
                self.gimbal.zoom_out()
                time.sleep(ZOOM_PULSE_ON)
                self.gimbal.zoom_stop()
            else:
                self.gimbal.zoom_in()
                time.sleep(ZOOM_PULSE_ON)
                self.gimbal.zoom_stop()
            self.current_zoom = target
            print(f'[Zoom] 搜索zoom切换到 {target}x')

    def zoom_to_max(self):
        """搜索开始时zoom到最大"""
        self.target_zoom = 6
        self.current_zoom = 6
        if self.gimbal:
            self.gimbal.zoom_in()
            time.sleep(3.0)
            self.gimbal.zoom_stop()
        print('[Zoom] 已zoom到最大(6x)')

    def zoom_to_min(self):
        """丢失恢复时zoom到最小"""
        self.target_zoom = 1
        if self.gimbal:
            self.gimbal.zoom_out()
            time.sleep(3.0)
            self.gimbal.zoom_stop()
        self.current_zoom = 1
        print('[Zoom] 已zoom到最小(1x)')

    def stop(self):
        if self.gimbal and self.zooming:
            self.gimbal.zoom_stop()
            self.zooming = False
