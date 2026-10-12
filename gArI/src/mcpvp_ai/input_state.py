from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from pynput import keyboard, mouse


@dataclass
class LiveInputState:
    watched_keys: list[str]
    pressed: set[str] = field(default_factory=set)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    left_click: bool = False
    right_click: bool = False
    hotbar: int | None = None
    recording: bool = False
    quit: bool = False
    emergency_stop: bool = False
    _last_pos: tuple[int, int] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot_and_reset_motion(self) -> dict:
        with self._lock:
            data = {
                "keys": {key: key in self.pressed for key in self.watched_keys},
                "mouse_dx": self.mouse_dx,
                "mouse_dy": self.mouse_dy,
                "left_click": self.left_click,
                "right_click": self.right_click,
                "hotbar": self.hotbar,
                "timestamp": time.time(),
            }
            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
            self.left_click = False
            self.right_click = False
            self.hotbar = None
            return data

    def set_key(self, key_name: str, active: bool) -> None:
        with self._lock:
            if active:
                self.pressed.add(key_name)
            else:
                self.pressed.discard(key_name)


def normalize_key(key: keyboard.Key | keyboard.KeyCode) -> str | None:
    if isinstance(key, keyboard.KeyCode) and key.char:
        char = key.char.lower()
        if char.isdigit() and "1" <= char <= "9":
            return char
        return char
    special = {
        keyboard.Key.space: "space",
        keyboard.Key.shift: "shift",
        keyboard.Key.shift_l: "shift",
        keyboard.Key.shift_r: "shift",
        keyboard.Key.ctrl: "ctrl",
        keyboard.Key.ctrl_l: "ctrl",
        keyboard.Key.ctrl_r: "ctrl",
        keyboard.Key.f8: "f8",
        keyboard.Key.f9: "f9",
        keyboard.Key.f10: "f10",
    }
    return special.get(key)


class InputListeners:
    def __init__(self, state: LiveInputState) -> None:
        self.state = state
        self.keyboard_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.mouse_listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click)

    def start(self) -> None:
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def stop(self) -> None:
        self.keyboard_listener.stop()
        self.mouse_listener.stop()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        name = normalize_key(key)
        if name == "f8":
            self.state.recording = not self.state.recording
            print(f"Recording: {self.state.recording}")
            return
        if name == "f9":
            self.state.quit = True
            return
        if name == "f10":
            self.state.emergency_stop = True
            self.state.quit = True
            return
        if name and name in self.state.watched_keys:
            self.state.set_key(name, True)
        if name and name.isdigit() and "1" <= name <= "9":
            with self.state._lock:
                self.state.hotbar = int(name) - 1

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        name = normalize_key(key)
        if name and name in self.state.watched_keys:
            self.state.set_key(name, False)

    def _on_move(self, x: int, y: int) -> None:
        with self.state._lock:
            if self.state._last_pos is not None:
                last_x, last_y = self.state._last_pos
                self.state.mouse_dx += x - last_x
                self.state.mouse_dy += y - last_y
            self.state._last_pos = (x, y)

    def _on_click(self, _x: int, _y: int, button: mouse.Button, pressed: bool) -> None:
        if not pressed:
            return
        with self.state._lock:
            if button == mouse.Button.left:
                self.state.left_click = True
            elif button == mouse.Button.right:
                self.state.right_click = True
