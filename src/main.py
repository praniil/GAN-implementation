"""Train a small fully-connected GAN on MNIST.

Besides the normal epoch sample grids, this script records enough evidence for
the README visualizations: fixed latent vectors, milestone samples, both final
model checkpoints, and epoch-average losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_RUN_DIR = PROJECT_ROOT / "generated" / "latest"

LATENT_DIM = 100
IMAGE_SIZE = 28 * 28
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
LEARNING_RATE = 0.001
MILESTONE_EPOCHS = {1, 10, 20, 30, 40, 50}


class Generator(nn.Module):
    """Map a 100-value latent vector to a 28 x 28 grayscale image."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, IMAGE_SIZE),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """Estimate the probability that an image came from MNIST."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMAGE_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image.view(-1, IMAGE_SIZE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optionally limit batches per epoch for a quick smoke test.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset() -> torchvision.datasets.MNIST:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return torchvision.datasets.MNIST(
        root=DATASET_DIR,
        train=True,
        transform=transform,
        download=True,
    )


def save_grid(images: torch.Tensor, destination: Path) -> None:
    grid = vutils.make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(destination, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close()


def train(
    *,
    epochs: int,
    seed: int,
    run_dir: Path,
    max_batches: int | None = None,
) -> None:
    if epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("--max-batches must be at least 1")

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = run_dir.resolve()
    samples_dir = run_dir / "samples"
    milestones_dir = run_dir / "milestones"
    samples_dir.mkdir(parents=True, exist_ok=True)
    milestones_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )

    evidence_rng = torch.Generator().manual_seed(seed + 1)
    fixed_noise = torch.randn(64, LATENT_DIM, generator=evidence_rng)
    analysis_noise = torch.randn(256, LATENT_DIM, generator=evidence_rng)
    real_samples = torch.stack([dataset[index][0] for index in range(64)])
    torch.save(fixed_noise, run_dir / "fixed_noise.pt")
    torch.save(analysis_noise, run_dir / "analysis_noise.pt")
    torch.save(real_samples, run_dir / "real_samples.pt")

    metrics: list[dict[str, float | int]] = []
    selected_milestones = MILESTONE_EPOCHS | {epochs}
    print(f"Training on {device} for {epochs} epoch(s); output: {run_dir}")

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        discriminator_loss_sum = 0.0
        generator_loss_sum = 0.0
        batch_count = 0

        for batch_index, (real_images, _) in enumerate(train_loader, start=1):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)
            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            optimizer_d.zero_grad()
            real_output = discriminator(real_images)
            loss_real = criterion(real_output, real_labels)
            noise = torch.randn(current_batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            loss_fake = criterion(fake_output, fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            noise = torch.randn(current_batch_size, LATENT_DIM, device=device)
            generated_images = generator(noise)
            output = discriminator(generated_images)
            loss_g = criterion(output, real_labels)
            loss_g.backward()
            optimizer_g.step()

            discriminator_loss_sum += loss_d.item()
            generator_loss_sum += loss_g.item()
            batch_count += 1
            if max_batches is not None and batch_index >= max_batches:
                break

        average_d = discriminator_loss_sum / batch_count
        average_g = generator_loss_sum / batch_count
        metrics.append(
            {
                "epoch": epoch,
                "discriminator_loss": average_d,
                "generator_loss": average_g,
            }
        )
        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"Loss D: {average_d:.4f} Loss G: {average_g:.4f}"
        )

        generator.eval()
        with torch.no_grad():
            generated = generator(fixed_noise.to(device)).cpu()
        save_grid(generated, samples_dir / f"epoch_{epoch:03d}.png")
        if epoch in selected_milestones:
            torch.save(generated, milestones_dir / f"epoch_{epoch:03d}.pt")

    torch.save(generator.state_dict(), run_dir / "generator.pth")
    torch.save(discriminator.state_dict(), run_dir / "discriminator.pth")

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "discriminator_loss", "generator_loss"],
        )
        writer.writeheader()
        writer.writerows(metrics)

    config = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": DEFAULT_BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "device": str(device),
        "max_batches": max_batches,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print("Training complete")


def main() -> None:
    args = parse_args()
    train(
        epochs=args.epochs,
        seed=args.seed,
        run_dir=args.run_dir,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
