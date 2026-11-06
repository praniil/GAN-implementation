"""Build reproducible README figures from a completed GAN training run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from main import Discriminator, Generator


EXPECTED_MILESTONES = (1, 10, 20, 30, 40, 50)
ACCENT_BLUE = "#2563eb"
ACCENT_ORANGE = "#ea580c"
INK = "#172033"
MUTED = "#64748b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing training artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Created {path}")


def build_latent_figure(
    fixed_noise: torch.Tensor,
    final_samples: torch.Tensor,
    output_dir: Path,
) -> None:
    vector = fixed_noise[0].reshape(10, 10).numpy()
    digit = final_samples[0, 0].numpy()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.7),
        gridspec_kw={"width_ratios": (1.2, 1)},
    )
    heatmap = axes[0].imshow(vector, cmap="coolwarm", vmin=-3, vmax=3)
    axes[0].set_title("100 random values (shown as 10 × 10)", color=INK, pad=12)
    axes[0].set_xlabel("The generator receives this as one flat vector", color=MUTED)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    colorbar = figure.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Value", color=MUTED)

    axes[1].imshow(digit, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title("Generated image after epoch 50", color=INK, pad=12)
    axes[1].set_xlabel("28 × 28 pixels", color=MUTED)
    axes[1].axis("off")
    figure.suptitle("A latent vector is a point in number space—not a hidden image", color=INK)
    figure.tight_layout()
    save_figure(figure, output_dir / "latent-to-digit.png")


def build_progression_figure(run_dir: Path, output_dir: Path) -> None:
    milestone_samples = {
        epoch: load_tensor(run_dir / "milestones" / f"epoch_{epoch:03d}.pt")
        for epoch in EXPECTED_MILESTONES
    }
    figure, axes = plt.subplots(8, 6, figsize=(10, 13))
    for column, epoch in enumerate(EXPECTED_MILESTONES):
        images = milestone_samples[epoch]
        for row in range(8):
            axis = axes[row, column]
            axis.imshow(images[row, 0], cmap="gray", vmin=-1, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"Epoch {epoch}", color=INK, fontsize=11)
            if column == 0:
                axis.set_ylabel(f"z{row + 1}", color=MUTED, rotation=0, labelpad=14)
    figure.suptitle(
        "The same eight latent vectors throughout training",
        color=INK,
        fontsize=16,
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "Because the inputs stay fixed, each row shows how one generated sample improves.",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.06, hspace=0.13, top=0.965, bottom=0.035)
    save_figure(figure, output_dir / "training-progression.png")


def build_discriminator_figure(run_dir: Path, output_dir: Path) -> None:
    generator = Generator()
    discriminator = Discriminator()
    generator.load_state_dict(load_tensor(run_dir / "generator.pth"))
    discriminator.load_state_dict(load_tensor(run_dir / "discriminator.pth"))
    generator.eval()
    discriminator.eval()

    analysis_noise = load_tensor(run_dir / "analysis_noise.pt")
    real_samples = load_tensor(run_dir / "real_samples.pt")
    with torch.no_grad():
        fake_samples = generator(analysis_noise)
        fake_scores = discriminator(fake_samples).flatten()
        real_scores = discriminator(real_samples).flatten()

    low_indices = torch.argsort(fake_scores)[:8]
    high_indices = torch.argsort(fake_scores, descending=True)[:8]
    rows = (
        ("Low-scoring fakes", fake_samples[low_indices], fake_scores[low_indices]),
        ("High-scoring fakes", fake_samples[high_indices], fake_scores[high_indices]),
        ("Real MNIST examples", real_samples[:8], real_scores[:8]),
    )

    figure, axes = plt.subplots(3, 8, figsize=(13.2, 5.8))
    for row_index, (label, images, scores) in enumerate(rows):
        for column in range(8):
            axis = axes[row_index, column]
            axis.imshow(images[column, 0], cmap="gray", vmin=-1, vmax=1)
            axis.set_title(f"D(x) {scores[column].item():.3f}", fontsize=8, color=MUTED)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(label, color=INK, fontsize=10, labelpad=12)
    figure.suptitle(
        "What the final discriminator scores as more or less real",
        color=INK,
        fontsize=16,
    )
    figure.text(
        0.5,
        0.015,
        "D(x) is a continuous score used for learning; the model does not delete images.",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.12, hspace=0.38, top=0.88, bottom=0.10)
    save_figure(figure, output_dir / "discriminator-scores.png")


def build_loss_figure(run_dir: Path, output_dir: Path) -> None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing training artifact: {metrics_path}")
    with metrics_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    epochs = np.array([int(row["epoch"]) for row in rows])
    discriminator_loss = np.array([float(row["discriminator_loss"]) for row in rows])
    generator_loss = np.array([float(row["generator_loss"]) for row in rows])

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(epochs, discriminator_loss, color=ACCENT_BLUE, label="Discriminator loss", linewidth=2)
    axis.plot(epochs, generator_loss, color=ACCENT_ORANGE, label="Generator loss", linewidth=2)
    axis.set_title("Training losses by epoch", color=INK, pad=12)
    axis.set_xlabel("Epoch", color=MUTED)
    axis.set_ylabel("Binary cross-entropy", color=MUTED)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, output_dir / "loss-curves.png")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_noise = load_tensor(run_dir / "fixed_noise.pt")
    final_samples = load_tensor(run_dir / "milestones" / "epoch_050.pt")
    build_latent_figure(fixed_noise, final_samples, output_dir)
    build_progression_figure(run_dir, output_dir)
    build_discriminator_figure(run_dir, output_dir)
    build_loss_figure(run_dir, output_dir)


if __name__ == "__main__":
    main()
