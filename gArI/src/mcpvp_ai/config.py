from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    root: Path

    @property
    def capture(self) -> dict[str, Any]:
        return self.raw["capture"]

    @property
    def actions(self) -> dict[str, Any]:
        return self.raw["actions"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    def path(self, key: str) -> Path:
        return (self.root / self.raw["paths"][key]).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return ProjectConfig(raw=raw, root=config_path.parents[1])
