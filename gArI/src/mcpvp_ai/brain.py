from __future__ import annotations

import argparse
import math
import time

import cv2
import torch

from .actions import Action, ActionController
from .config import load_config
from .input_state import InputListeners, LiveInputState
from .model import VisionPolicy
from .screen import ScreenCapture
from .targeter import make_targeter


def image_to_tensor(image_bgr, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def decode_action(outputs: dict[str, torch.Tensor], keys: list[str], action_config: dict) -> Action:
    key_probs = torch.sigmoid(outputs["keys"][0]).detach().cpu()
    click_probs = torch.sigmoid(outputs["clicks"][0]).detach().cpu()
    mouse = outputs["mouse"][0].detach().cpu()
    hotbar = int(torch.argmax(outputs["hotbar"][0]).item())
    default_key_threshold = float(action_config.get("key_threshold", 0.55))
    per_key_thresholds = action_config.get("per_key_thresholds", {})
    click_threshold = float(action_config.get("click_threshold", 0.65))
    max_mouse_delta = float(action_config["max_mouse_delta"])

    return Action(
        keys={
            key: bool(key_probs[i] > float(per_key_thresholds.get(key, default_key_threshold)))
            for i, key in enumerate(keys)
        },
        mouse_dx=float(mouse[0]) * max_mouse_delta,
        mouse_dy=float(mouse[1]) * max_mouse_delta,
        left_click=bool(click_probs[0] > click_threshold),
        right_click=bool(click_probs[1] > click_threshold),
        hotbar=None if hotbar == 9 else hotbar,
    )


class ActionSmoother:
    def __init__(self, keys: list[str], hold_frames: int) -> None:
        self.remaining = {key: 0 for key in keys}
        self.hold_frames = max(1, hold_frames)

    def smooth(self, action: Action) -> Action:
        smoothed_keys: dict[str, bool] = {}
        for key, active in action.keys.items():
            if active:
                self.remaining[key] = self.hold_frames
            elif self.remaining[key] > 0:
                self.remaining[key] -= 1
            smoothed_keys[key] = self.remaining[key] > 0

        action.keys = smoothed_keys
        return action


class MouseSmoother:
    def __init__(self, action_config: dict) -> None:
        self.smoothing = min(0.95, max(0.0, float(action_config.get("mouse_smoothing", 0.35))))
        self.sensitivity = float(action_config.get("mouse_sensitivity", 1.0))
        self.min_pixels = float(action_config.get("mouse_min_pixels", 1.0))
        self.max_step = float(action_config.get("mouse_max_step", 45.0))
        self.last_dx = 0.0
        self.last_dy = 0.0

    def smooth(self, action: Action) -> Action:
        target_dx = action.mouse_dx * self.sensitivity
        target_dy = action.mouse_dy * self.sensitivity
        dx = self.last_dx * self.smoothing + target_dx * (1.0 - self.smoothing)
        dy = self.last_dy * self.smoothing + target_dy * (1.0 - self.smoothing)

        if abs(dx) < self.min_pixels:
            dx = 0.0
        if abs(dy) < self.min_pixels:
            dy = 0.0

        action.mouse_dx = max(-self.max_step, min(self.max_step, dx))
        action.mouse_dy = max(-self.max_step, min(self.max_step, dy))
        self.last_dx = action.mouse_dx
        self.last_dy = action.mouse_dy
        return action


class AimBeforeMove:
    def __init__(self, action_config: dict) -> None:
        self.enabled = bool(action_config.get("aim_before_move", True))
        self.threshold = float(action_config.get("aim_move_block_threshold", 6.0))
        self.release_keys = set(action_config.get("aim_move_release_keys", ["w", "a", "s", "d", "ctrl"]))

    def filter(self, action: Action) -> Action:
        if not self.enabled:
            return action

        turn_amount = math.hypot(action.mouse_dx, action.mouse_dy)
        if turn_amount < self.threshold:
            return action

        for key in self.release_keys:
            if key in action.keys:
                action.keys[key] = False
        return action


class TargetAimAssist:
    def __init__(self, config: dict, frame_width: int, frame_height: int) -> None:
        self.config = config
        self.targeter = make_targeter(config, frame_width, frame_height)
        self.override_mouse = bool(config.get("override_brain_mouse", True))
        self.block_movement = bool(config.get("block_movement_until_targeted", True))
        self.targeted_deadzone = float(config.get("targeted_deadzone_px", 10))
        self.release_keys = {"w", "a", "s", "d", "ctrl"}

    def apply(self, action: Action, image_bgr) -> tuple[Action, bool]:
        target = self.targeter.find(image_bgr)
        if target is None:
            return action, False

        dx, dy = self.targeter.aim_delta(target)
        if self.override_mouse:
            action.mouse_dx = dx
            action.mouse_dy = dy

        if self.block_movement and math.hypot(dx, dy) > self.targeted_deadzone:
            for key in self.release_keys:
                if key in action.keys:
                    action.keys[key] = False

        return action, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--control", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = cfg.path("latest_checkpoint")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Record data and train behavior cloning first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    keys = checkpoint.get("keys", cfg.actions["keys"])
    model = VisionPolicy(num_keys=len(keys)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    state = LiveInputState(watched_keys=keys)
    listeners = InputListeners(state)
    capture = ScreenCapture(
        monitor_index=int(cfg.capture["monitor_index"]),
        width=int(cfg.capture["width"]),
        height=int(cfg.capture["height"]),
        window_title=cfg.capture.get("window_title"),
    )
    controller = ActionController(keys)
    smoother = ActionSmoother(keys, int(cfg.actions.get("key_hold_frames", 4)))
    mouse_smoother = MouseSmoother(cfg.actions)
    aim_filter = AimBeforeMove(cfg.actions)
    target_aim = TargetAimAssist(
        cfg.raw.get("targeting", {}),
        frame_width=int(cfg.capture["width"]),
        frame_height=int(cfg.capture["height"]),
    )
    frame_interval = 1.0 / float(cfg.capture["fps"])

    print("Brain running. Press F10 for emergency stop.")
    print("Mode:", "control" if args.control else "dry-run")
    listeners.start()
    try:
        while not state.quit:
            start = time.time()
            frame = capture.grab()
            with torch.no_grad():
                outputs = model(image_to_tensor(frame.image_bgr, device))
            action = decode_action(outputs, keys, cfg.actions)
            action = mouse_smoother.smooth(action)
            action = smoother.smooth(action)
            action, target_found = target_aim.apply(action, frame.image_bgr)
            action = aim_filter.filter(action)

            if args.control:
                controller.apply(action)
            else:
                active_keys = [key for key, active in action.keys.items() if active]
                print(
                    f"keys={active_keys} mouse=({action.mouse_dx:.1f},{action.mouse_dy:.1f}) "
                    f"left={action.left_click} right={action.right_click} hotbar={action.hotbar} target={target_found}"
                )

            elapsed = time.time() - start
            time.sleep(max(0.0, frame_interval - elapsed))
    finally:
        controller.release_all()
        listeners.stop()
        capture.close()
        print("Brain stopped.")


if __name__ == "__main__":
    main()
