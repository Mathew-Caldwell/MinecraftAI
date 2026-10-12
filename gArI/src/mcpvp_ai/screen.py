from __future__ import annotations

import ctypes
import ctypes.wintypes
import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass
class Frame:
    image_bgr: np.ndarray
    timestamp: float


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


user32 = ctypes.windll.user32


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    return rect.left, rect.top, width, height


def list_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []

    enum_proc_type = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _window_title(hwnd).strip()
        rect = _window_rect(hwnd)
        if title and rect:
            left, top, width, height = rect
            windows.append(WindowInfo(hwnd=hwnd, title=title, left=left, top=top, width=width, height=height))
        return True

    user32.EnumWindows(enum_proc_type(callback), 0)
    return windows


def find_window(title_contains: str) -> WindowInfo | None:
    needle = title_contains.lower()
    matches = [window for window in list_windows() if needle in window.title.lower()]
    if not matches:
        return None
    matches.sort(key=lambda window: window.width * window.height, reverse=True)
    return matches[0]


class ScreenCapture:
    def __init__(self, monitor_index: int, width: int, height: int, window_title: str | None = None) -> None:
        self.sct = mss.mss()
        self.monitor_index = monitor_index
        self.window_title = window_title
        self.width = width
        self.height = height
        self.monitor = self._resolve_capture_region()

    def _resolve_capture_region(self) -> dict:
        if self.window_title:
            window = find_window(self.window_title)
            if window:
                return {
                    "left": window.left,
                    "top": window.top,
                    "width": window.width,
                    "height": window.height,
                }
            raise RuntimeError(
                f"No visible window found with title containing {self.window_title!r}. "
                "Run scripts\\list_windows.bat and update capture.window_title in configs\\default.json."
            )
        return self.sct.monitors[self.monitor_index]

    def grab(self) -> Frame:
        self.monitor = self._resolve_capture_region()
        raw = np.array(self.sct.grab(self.monitor))
        bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        resized = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return Frame(image_bgr=resized, timestamp=time.time())

    def close(self) -> None:
        self.sct.close()


def list_monitors() -> list[dict]:
    with mss.mss() as sct:
        return [dict(monitor) for monitor in sct.monitors]
