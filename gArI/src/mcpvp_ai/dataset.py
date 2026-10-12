from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PvPDataset(Dataset):
    def __init__(self, recordings_dir: Path, keys: list[str], width: int, height: int, max_mouse_delta: float) -> None:
        self.keys = keys
        self.width = width
        self.height = height
        self.max_mouse_delta = max_mouse_delta
        self.samples: list[tuple[Path, dict]] = []

        for actions_path in sorted(recordings_dir.glob("*/actions.jsonl")):
            frames_dir = actions_path.parent / "frames"
            with actions_path.open("r", encoding="utf-8") as f:
                for line in f:
                    action = json.loads(line)
                    frame_path = frames_dir / action["frame"]
                    if frame_path.exists():
                        self.samples.append((frame_path, action))

    def __len__(self) -> int:
        return len(self.samples)

    def action_rates(self) -> dict[str, float]:
        if not self.samples:
            return {}

        counts = {key: 0 for key in self.keys}
        click_count = 0
        mouse_count = 0
        for _frame_path, action in self.samples:
            for key in self.keys:
                counts[key] += int(bool(action["keys"].get(key, False)))
            click_count += int(bool(action.get("left_click", False)) or bool(action.get("right_click", False)))
            mouse_count += int(abs(float(action.get("mouse_dx", 0.0))) > 1 or abs(float(action.get("mouse_dy", 0.0))) > 1)

        total = float(len(self.samples))
        return {
            **{key: counts[key] / total for key in self.keys},
            "click_any": click_count / total,
            "mouse_moved": mouse_count / total,
        }

    def sample_weights(self) -> torch.Tensor:
        weights: list[float] = []
        for _frame_path, action in self.samples:
            keys = action["keys"]
            weight = 1.0
            if not bool(keys.get("w", False)):
                weight += 2.0
            if any(bool(keys.get(key, False)) for key in ("a", "s", "d", "space", "shift", "ctrl")):
                weight += 2.0
            if bool(action.get("left_click", False)) or bool(action.get("right_click", False)):
                weight += 2.0
            if abs(float(action.get("mouse_dx", 0.0))) > 1 or abs(float(action.get("mouse_dy", 0.0))) > 1:
                weight += 1.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.double)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame_path, action = self.samples[index]
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame_path)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        key_targets = torch.tensor([float(action["keys"].get(key, False)) for key in self.keys], dtype=torch.float32)
        mouse = torch.tensor(
            [
                float(action.get("mouse_dx", 0.0)) / self.max_mouse_delta,
                float(action.get("mouse_dy", 0.0)) / self.max_mouse_delta,
            ],
            dtype=torch.float32,
        ).clamp(-1.0, 1.0)
        clicks = torch.tensor(
            [float(action.get("left_click", False)), float(action.get("right_click", False))],
            dtype=torch.float32,
        )
        hotbar_value = action.get("hotbar")
        hotbar = torch.tensor(9 if hotbar_value is None else int(hotbar_value), dtype=torch.long)

        return {"image": image, "keys": key_targets, "mouse": mouse, "clicks": clicks, "hotbar": hotbar}
