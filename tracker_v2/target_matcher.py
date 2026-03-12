"""
多特征融合匹配器 — 颜色+形状+YOLO综合判断"是不是那个目标"
"""

import cv2
import numpy as np
from config import (
    WEIGHT_COLOR, WEIGHT_SHAPE, WEIGHT_YOLO,
    MATCH_THRESHOLD, VERIFY_THRESHOLD,
    HSV_HIST_SIZE, HSV_RANGES, TEMPLATE_SCALES,
)


class TargetMatcher:
    """多特征融合目标匹配"""

    def __init__(self, feature_bank):
        self.bank = feature_bank

    def match_crop(self, crop, yolo_conf=0.0):
        """
        对一个候选裁剪区域，与特征库中所有模板比对，返回最高综合分

        Args:
            crop: BGR图像裁剪
            yolo_conf: YOLO检测置信度 (0-1)

        Returns:
            (score, best_template_idx) 或 (0.0, -1) 如果不匹配
        """
        if crop is None or crop.size == 0 or not self.bank.is_loaded():
            return 0.0, -1

        # 提取候选目标的特征
        color_score = self._best_color_match(crop)
        shape_score = self._best_shape_match(crop)

        # 综合评分
        score = (WEIGHT_COLOR * color_score +
                 WEIGHT_SHAPE * shape_score +
                 WEIGHT_YOLO * yolo_conf)

        return score, 0  # 简化: 返回最佳模板index

    def match_detections(self, frame, detections, threshold=None):
        """
        对YOLO检测到的所有候选目标，找出最匹配照片集的那个

        Args:
            frame: 完整帧
            detections: [(x1,y1,x2,y2,conf,cls), ...]
            threshold: 匹配阈值，默认用 MATCH_THRESHOLD

        Returns:
            (best_det, score) 或 (None, 0.0)
        """
        if threshold is None:
            threshold = MATCH_THRESHOLD

        best_det = None
        best_score = 0.0

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            score, _ = self.match_crop(crop, yolo_conf=conf)
            if score > best_score:
                best_score = score
                best_det = det

        if best_score >= threshold:
            return best_det, best_score
        return None, 0.0

    def verify_target(self, crop):
        """
        追踪中持续验证：当前追踪的还是原目标吗？

        Returns:
            True 如果仍然匹配，False 如果可能跟错了
        """
        score, _ = self.match_crop(crop)
        return score >= VERIFY_THRESHOLD

    def template_search(self, frame):
        """
        无YOLO检测结果时，纯模板匹配搜索全画面

        Returns:
            (x1, y1, x2, y2, score) 或 None
        """
        if not self.bank.is_loaded():
            return None

        # 降采样搜索: 先在半分辨率上找粗位置，大幅减少耗时
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        best_result = None
        best_score = 0.0

        # 只用3个关键尺度，不遍历全部
        search_scales = [0.5, 0.75, 1.0]

        for template in self.bank.templates:
            for scale in search_scales:
                if scale not in template['multi_scale']:
                    continue
                tmpl_gray = template['multi_scale'][scale]
                # 模板也降采样一半
                th, tw = tmpl_gray.shape[:2]
                stw, sth = tw // 2, th // 2
                if stw < 4 or sth < 4:
                    continue
                small_tmpl = cv2.resize(tmpl_gray, (stw, sth))

                if stw >= gray.shape[1] or sth >= gray.shape[0]:
                    continue

                result = cv2.matchTemplate(gray, small_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    mx, my = max_loc
                    # 坐标映射回原分辨率
                    best_result = (mx * 2, my * 2, mx * 2 + tw, my * 2 + th, max_val)

        if best_result and best_score > 0.5:
            return best_result
        return None

    def _best_color_match(self, crop):
        """与特征库中所有模板做颜色直方图比较，返回最高分(0-1)"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, HSV_HIST_SIZE, HSV_RANGES)
        cv2.normalize(hist, hist)

        best = 0.0
        for tmpl_hist in self.bank.get_histograms():
            score = cv2.compareHist(hist, tmpl_hist, cv2.HISTCMP_CORREL)
            # CORREL范围-1~1，映射到0~1
            score = max(0.0, (score + 1.0) / 2.0)
            best = max(best, score)

        return best

    def _best_shape_match(self, crop):
        """与特征库中所有模板做Hu矩比较，返回最高分(0-1)"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        best = 0.0
        for tmpl_hu in self.bank.get_hu_moments():
            # 欧氏距离，转换为相似度
            dist = np.linalg.norm(hu - tmpl_hu)
            score = 1.0 / (1.0 + dist * 0.1)
            best = max(best, score)

        return best
