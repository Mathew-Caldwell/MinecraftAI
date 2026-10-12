from __future__ import annotations

import torch
from torch import nn


class VisionPolicy(nn.Module):
    def __init__(self, num_keys: int, hotbar_classes: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
        )
        self.key_head = nn.Linear(512, num_keys)
        self.mouse_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 2), nn.Tanh())
        self.click_head = nn.Linear(512, 2)
        self.hotbar_head = nn.Linear(512, hotbar_classes)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        return {
            "keys": self.key_head(features),
            "mouse": self.mouse_head(features),
            "clicks": self.click_head(features),
            "hotbar": self.hotbar_head(features),
        }
