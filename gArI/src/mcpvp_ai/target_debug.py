from __future__ import annotations

import argparse
import time

import cv2

from .config import load_config
from .input_state import InputListeners, LiveInputState
from .screen import ScreenCapture
from .targeter import make_targeter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    capture = ScreenCapture(
        monitor_index=int(cfg.capture["monitor_index"]),
        width=int(cfg.capture["width"]),
        height=int(cfg.capture["height"]),
        window_title=cfg.capture.get("window_title"),
    )
    targeter = make_targeter(
        cfg.raw.get("targeting", {}),
        frame_width=int(cfg.capture["width"]),
        frame_height=int(cfg.capture["height"]),
    )
    state = LiveInputState(watched_keys=cfg.actions["keys"])
    listeners = InputListeners(state)
    listeners.start()

    print("Target debug running. Press F10 to quit.")
    try:
        while not state.quit:
            frame = capture.grab()
            target = targeter.find(frame.image_bgr)
            preview = targeter.annotate(frame.image_bgr, target)
            cv2.imshow("Minecraft PvP Targeter", preview)
            cv2.waitKey(1)
            if target:
                dx, dy = targeter.aim_delta(target)
                print(f"target=({target.center_x:.1f},{target.center_y:.1f}) aim_mouse=({dx:.1f},{dy:.1f})")
            time.sleep(1.0 / float(cfg.capture["fps"]))
    finally:
        listeners.stop()
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
