from __future__ import annotations

from .screen import list_monitors


def main() -> None:
    print("Available capture monitors:")
    for index, monitor in enumerate(list_monitors()):
        label = "all monitors combined" if index == 0 else f"monitor {index}"
        print(
            f"{index}: {label} "
            f"left={monitor['left']} top={monitor['top']} "
            f"width={monitor['width']} height={monitor['height']}"
        )
    print()
    print("Use the number for the monitor Minecraft is on as capture.monitor_index in configs\\default.json.")


if __name__ == "__main__":
    main()
