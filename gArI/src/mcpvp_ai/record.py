from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .config import load_config
from .input_state import InputListeners, LiveInputState
from .screen import ScreenCapture


def make_session_dir(recordings_dir: Path) -> Path:
    session = recordings_dir / time.strftime("%Y%m%d_%H%M%S")
    (session / "frames").mkdir(parents=True, exist_ok=True)
    return session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    recordings_dir = cfg.path("recordings_dir")
    recordings_dir.mkdir(parents=True, exist_ok=True)
    session = make_session_dir(recordings_dir)

    state = LiveInputState(watched_keys=cfg.actions["keys"])
    listeners = InputListeners(state)
    capture = ScreenCapture(
        monitor_index=int(cfg.capture["monitor_index"]),
        width=int(cfg.capture["width"]),
        height=int(cfg.capture["height"]),
        window_title=cfg.capture.get("window_title"),
    )
    frame_interval = 1.0 / float(cfg.capture["fps"])

    print("Recorder ready. Focus Minecraft, press F8 to start/stop, F9 to quit.")
    print(f"Saving to: {session}")
    listeners.start()
    frame_id = 0

    try:
        with (session / "actions.jsonl").open("a", encoding="utf-8") as action_file:
            while not state.quit:
                start = time.time()
                if state.recording:
                    frame = capture.grab()
                    action = state.snapshot_and_reset_motion()
                    frame_name = f"{frame_id:08d}.jpg"
                    cv2.imwrite(str(session / "frames" / frame_name), frame.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    action["frame"] = frame_name
                    action["frame_timestamp"] = frame.timestamp
                    action_file.write(json.dumps(action) + "\n")
                    frame_id += 1
                elapsed = time.time() - start
                time.sleep(max(0.0, frame_interval - elapsed))
    finally:
        listeners.stop()
        capture.close()
        print(f"Recorded {frame_id} frames.")


if __name__ == "__main__":
    main()
