from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from tqdm import tqdm

from .config import load_config
from .dataset import PvPDataset
from .model import VisionPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PvPDataset(
        recordings_dir=cfg.path("recordings_dir"),
        keys=cfg.actions["keys"],
        width=int(cfg.capture["width"]),
        height=int(cfg.capture["height"]),
        max_mouse_delta=float(cfg.actions["max_mouse_delta"]),
    )
    if len(dataset) < 10:
        raise RuntimeError("Not enough recorded frames yet. Record at least a few minutes of gameplay first.")

    rates = dataset.action_rates()
    print("Recorded action rates:")
    for name, rate in rates.items():
        print(f"  {name}: {rate * 100:.1f}%")

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    sampler = None
    shuffle = True
    if bool(cfg.training.get("balanced_sampling", True)):
        all_weights = dataset.sample_weights()
        train_indices = train_set.indices if isinstance(train_set, Subset) else list(range(len(train_set)))
        train_weights = all_weights[train_indices]
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.training["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.training["num_workers"]),
    )
    val_loader = DataLoader(val_set, batch_size=int(cfg.training["batch_size"]), shuffle=False)

    model = VisionPolicy(num_keys=len(cfg.actions["keys"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training["learning_rate"]))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    for epoch in range(int(cfg.training["epochs"])):
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1} train"):
            image = batch["image"].to(device)
            outputs = model(image)
            loss = (
                bce(outputs["keys"], batch["keys"].to(device))
                + mse(outputs["mouse"], batch["mouse"].to(device))
                + bce(outputs["clicks"], batch["clicks"].to(device))
                + ce(outputs["hotbar"], batch["hotbar"].to(device))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                outputs = model(image)
                loss = (
                    bce(outputs["keys"], batch["keys"].to(device))
                    + mse(outputs["mouse"], batch["mouse"].to(device))
                    + bce(outputs["clicks"], batch["clicks"].to(device))
                    + ce(outputs["hotbar"], batch["hotbar"].to(device))
                )
                val_total += loss.item()

        print(
            f"epoch={epoch + 1} train_loss={total / max(1, len(train_loader)):.4f} "
            f"val_loss={val_total / max(1, len(val_loader)):.4f}"
        )

    checkpoint_path = cfg.path("latest_checkpoint")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "keys": cfg.actions["keys"],
            "config": cfg.raw,
        },
        checkpoint_path,
    )
    print(f"Saved brain to {checkpoint_path}")


if __name__ == "__main__":
    main()
