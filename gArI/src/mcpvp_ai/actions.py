from __future__ import annotations

from dataclasses import dataclass, field

import pydirectinput


@dataclass
class Action:
    keys: dict[str, bool] = field(default_factory=dict)
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    left_click: bool = False
    right_click: bool = False
    hotbar: int | None = None

    def to_json(self) -> dict:
        return {
            "keys": self.keys,
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "left_click": self.left_click,
            "right_click": self.right_click,
            "hotbar": self.hotbar,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Action":
        return cls(
            keys=dict(data.get("keys", {})),
            mouse_dx=float(data.get("mouse_dx", 0.0)),
            mouse_dy=float(data.get("mouse_dy", 0.0)),
            left_click=bool(data.get("left_click", False)),
            right_click=bool(data.get("right_click", False)),
            hotbar=data.get("hotbar"),
        )


class ActionController:
    def __init__(self, allowed_keys: list[str]) -> None:
        self.allowed_keys = allowed_keys
        self.held: set[str] = set()
        pydirectinput.PAUSE = 0

    def release_all(self) -> None:
        for key in list(self.held):
            pydirectinput.keyUp(key)
        self.held.clear()

    def apply(self, action: Action) -> None:
        desired = {key for key, active in action.keys.items() if active and key in self.allowed_keys}

        for key in self.held - desired:
            pydirectinput.keyUp(key)
        for key in desired - self.held:
            pydirectinput.keyDown(key)
        self.held = desired

        if abs(action.mouse_dx) >= 1 or abs(action.mouse_dy) >= 1:
            pydirectinput.moveRel(int(action.mouse_dx), int(action.mouse_dy), duration=0.02)

        if action.left_click:
            pydirectinput.click(button="left")
        if action.right_click:
            pydirectinput.click(button="right")
        if action.hotbar is not None:
            pydirectinput.press(str(int(action.hotbar) + 1))
