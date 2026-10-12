from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Target:
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float


class ColorTargeter:
    def __init__(self, config: dict, frame_width: int, frame_height: int) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.hsv_lower = np.array(config.get("hsv_lower", [0, 80, 80]), dtype=np.uint8)
        self.hsv_upper = np.array(config.get("hsv_upper", [12, 255, 255]), dtype=np.uint8)
        self.min_area = float(config.get("min_area", 40))
        self.aim_y_ratio = float(config.get("aim_y_ratio", 0.35))
        self.deadzone_px = float(config.get("deadzone_px", 3))
        self.mouse_gain = float(config.get("mouse_gain", 0.22))
        self.max_mouse_step = float(config.get("max_mouse_step", 35))
        self.frame_width = frame_width
        self.frame_height = frame_height

    def find(self, image_bgr: np.ndarray) -> Target | None:
        if not self.enabled:
            return None

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        screen_center_x = self.frame_width / 2.0
        best: Target | None = None
        best_score = -1.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h < 6 or w < 3:
                continue

            center_x = x + w / 2.0
            aim_y = y + h * self.aim_y_ratio
            center_bias = 1.0 / (1.0 + abs(center_x - screen_center_x) / screen_center_x)
            score = area * center_bias
            if score > best_score:
                best_score = score
                best = Target(
                    center_x=center_x,
                    center_y=aim_y,
                    width=float(w),
                    height=float(h),
                    confidence=min(1.0, area / max(self.min_area * 8.0, 1.0)),
                )

        return best

    def aim_delta(self, target: Target) -> tuple[float, float]:
        error_x = target.center_x - self.frame_width / 2.0
        error_y = target.center_y - self.frame_height / 2.0

        if abs(error_x) < self.deadzone_px:
            error_x = 0.0
        if abs(error_y) < self.deadzone_px:
            error_y = 0.0

        dx = max(-self.max_mouse_step, min(self.max_mouse_step, error_x * self.mouse_gain))
        dy = max(-self.max_mouse_step, min(self.max_mouse_step, error_y * self.mouse_gain))
        return dx, dy

    def annotate(self, image_bgr: np.ndarray, target: Target | None) -> np.ndarray:
        output = image_bgr.copy()
        center = (self.frame_width // 2, self.frame_height // 2)
        cv2.drawMarker(output, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        if target is None:
            return output

        x1 = int(target.center_x - target.width / 2.0)
        y1 = int(target.center_y - target.height * self.aim_y_ratio)
        x2 = int(x1 + target.width)
        y2 = int(y1 + target.height)
        aim = (int(target.center_x), int(target.center_y))
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.drawMarker(output, aim, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        cv2.line(output, center, aim, (0, 255, 255), 1)
        return output


class MotionTargeter:
    def __init__(self, config: dict, frame_width: int, frame_height: int) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.previous_gray: np.ndarray | None = None
        self.min_area = float(config.get("motion_min_area", 90))
        self.threshold = int(config.get("motion_threshold", 18))
        self.aim_y_ratio = float(config.get("aim_y_ratio", 0.35))
        self.deadzone_px = float(config.get("deadzone_px", 3))
        self.mouse_gain = float(config.get("mouse_gain", 0.22))
        self.max_mouse_step = float(config.get("max_mouse_step", 35))
        self.center_bias_strength = float(config.get("center_bias_strength", 1.7))
        self.ignore_bottom_ratio = float(config.get("ignore_bottom_ratio", 0.22))
        self.ignore_right_ratio = float(config.get("ignore_right_ratio", 0.18))

    def find(self, image_bgr: np.ndarray) -> Target | None:
        if not self.enabled:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.previous_gray is None:
            self.previous_gray = gray
            return None

        diff = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray
        _unused, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        ignore_bottom_y = int(self.frame_height * (1.0 - self.ignore_bottom_ratio))
        ignore_right_x = int(self.frame_width * (1.0 - self.ignore_right_ratio))
        mask[ignore_bottom_y:, :] = 0
        mask[:, ignore_right_x:] = 0

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        screen_center_x = self.frame_width / 2.0
        best: Target | None = None
        best_score = -1.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h < 14 or w < 5:
                continue

            aspect = h / max(w, 1)
            if aspect < 0.7:
                continue

            center_x = x + w / 2.0
            aim_y = y + h * self.aim_y_ratio
            center_distance = abs(center_x - screen_center_x) / screen_center_x
            center_bias = 1.0 / (1.0 + center_distance * self.center_bias_strength)
            vertical_bonus = min(2.0, aspect)
            score = area * center_bias * vertical_bonus
            if score > best_score:
                best_score = score
                best = Target(
                    center_x=center_x,
                    center_y=aim_y,
                    width=float(w),
                    height=float(h),
                    confidence=min(1.0, area / max(self.min_area * 8.0, 1.0)),
                )

        return best

    def aim_delta(self, target: Target) -> tuple[float, float]:
        error_x = target.center_x - self.frame_width / 2.0
        error_y = target.center_y - self.frame_height / 2.0

        if abs(error_x) < self.deadzone_px:
            error_x = 0.0
        if abs(error_y) < self.deadzone_px:
            error_y = 0.0

        dx = max(-self.max_mouse_step, min(self.max_mouse_step, error_x * self.mouse_gain))
        dy = max(-self.max_mouse_step, min(self.max_mouse_step, error_y * self.mouse_gain))
        return dx, dy

    def annotate(self, image_bgr: np.ndarray, target: Target | None) -> np.ndarray:
        output = image_bgr.copy()
        center = (self.frame_width // 2, self.frame_height // 2)
        cv2.drawMarker(output, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
        cv2.line(output, (0, int(self.frame_height * (1.0 - self.ignore_bottom_ratio))), (self.frame_width, int(self.frame_height * (1.0 - self.ignore_bottom_ratio))), (80, 80, 80), 1)
        if target is None:
            return output

        x1 = int(target.center_x - target.width / 2.0)
        y1 = int(target.center_y - target.height * self.aim_y_ratio)
        x2 = int(x1 + target.width)
        y2 = int(y1 + target.height)
        aim = (int(target.center_x), int(target.center_y))
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv2.drawMarker(output, aim, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        cv2.line(output, center, aim, (0, 255, 255), 1)
        return output


def make_targeter(config: dict, frame_width: int, frame_height: int):
    mode = str(config.get("mode", "motion")).lower()
    if mode == "color":
        return ColorTargeter(config, frame_width, frame_height)
    return MotionTargeter(config, frame_width, frame_height)
