
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_FOLDER = Path("test_images")
OUTPUT_FOLDER = Path("experiment_outputs/visual_reconstruction")
IMAGE_SIZE = (64, 64)

STEPS = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.005
SEED = 11


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class FourierMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Fixed random Fourier projection
        generator = torch.Generator().manual_seed(SEED)
        self.register_buffer(
            "B",
            torch.randn(2, 16, generator=generator) * 3.0
        )

        self.network = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, coordinates):
        projection = 2 * np.pi * coordinates @ self.B
        encoded = torch.cat(
            [torch.sin(projection), torch.cos(projection)],
            dim=1
        )
        return self.network(encoded)


def load_image(path):
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    original = np.asarray(image, dtype=np.float32) / 255.0

    height, width, _ = original.shape

    ys, xs = np.meshgrid(
        np.linspace(-1, 1, height, dtype=np.float32),
        np.linspace(-1, 1, width, dtype=np.float32),
        indexing="ij"
    )

    coords = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    colors = original.reshape(-1, 3)

    return original, torch.tensor(coords), torch.tensor(colors)


def train_and_reconstruct(model_class, X, Y):
    set_seed(SEED)

    model = model_class()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    loss_function = nn.MSELoss()

    for step in range(STEPS):
        indices = torch.randint(
            0, len(X), (BATCH_SIZE,)
        )

        predictions = model(X[indices])
        loss = loss_function(predictions, Y[indices])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        reconstructed = model(X).numpy().reshape(
            IMAGE_SIZE[1], IMAGE_SIZE[0], 3
        )

    return reconstructed


def calculate_metrics(original, reconstructed):
    mse = float(np.mean((original - reconstructed) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    return mse, psnr


def main():
    set_seed(SEED)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGE_FOLDER.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    )

    if not image_paths:
        print("No images found in test_images folder.")
        return

    rows = []

    for path in image_paths:
        print(f"\nProcessing: {path.name}", flush=True)

        original, X, Y = load_image(path)

        baseline = train_and_reconstruct(BaselineMLP, X, Y)
        fourier = train_and_reconstruct(FourierMLP, X, Y)

        baseline_mse, baseline_psnr = calculate_metrics(
            original, baseline
        )
        fourier_mse, fourier_psnr = calculate_metrics(
            original, fourier
        )

        reduction = (
            (baseline_mse - fourier_mse) / baseline_mse * 100
            if baseline_mse > 0 else 0
        )

        print(
            f"Baseline: MSE={baseline_mse:.6f}, "
            f"PSNR={baseline_psnr:.2f} dB"
        )
        print(
            f"Fourier:  MSE={fourier_mse:.6f}, "
            f"PSNR={fourier_psnr:.2f} dB"
        )
        print(f"MSE reduction: {reduction:.2f}%")

        # Save the three-image visual comparison
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(original)
        axes[0].set_title("Original")

        axes[1].imshow(baseline)
        axes[1].set_title(
            f"Baseline\nMSE={baseline_mse:.5f}"
        )

        axes[2].imshow(fourier)
        axes[2].set_title(
            f"Fourier Features\nMSE={fourier_mse:.5f}"
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(path.name)
        plt.tight_layout()

        output_path = OUTPUT_FOLDER / f"{path.stem}_comparison.png"
        plt.savefig(output_path, dpi=200)
        plt.close()

        # Save individual reconstructions
        Image.fromarray(
            np.uint8(np.clip(baseline * 255, 0, 255))
        ).save(OUTPUT_FOLDER / f"{path.stem}_baseline.png")

        Image.fromarray(
            np.uint8(np.clip(fourier * 255, 0, 255))
        ).save(OUTPUT_FOLDER / f"{path.stem}_fourier.png")

        rows.append({
            "Image": path.name,
            "Baseline_MSE": baseline_mse,
            "Baseline_PSNR_dB": baseline_psnr,
            "Fourier_MSE": fourier_mse,
            "Fourier_PSNR_dB": fourier_psnr,
            "MSE_Reduction_Percent": reduction
        })

    import csv

    csv_path = OUTPUT_FOLDER / "visual_comparison_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nVisual comparisons saved to:")
    print(OUTPUT_FOLDER.resolve())
    print("Metrics:", csv_path.resolve())


if __name__ == "__main__":
    main()
