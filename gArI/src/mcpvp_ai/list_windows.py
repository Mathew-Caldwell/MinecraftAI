from __future__ import annotations

from .screen import list_windows


def main() -> None:
    print("Visible windows:")
    for window in list_windows():
        print(
            f"hwnd={window.hwnd} "
            f"left={window.left} top={window.top} "
            f"width={window.width} height={window.height} "
            f"title={window.title}"
        )
    print()
    print("If Minecraft is listed, put part of its title in capture.window_title in configs\\default.json.")


if __name__ == "__main__":
    main()
