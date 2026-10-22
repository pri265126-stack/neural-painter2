
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


# ---------------- SETTINGS ----------------

IMAGE_PATHS = [
    Path("test_images/flower.jpg"),
    Path("test_images/place.jpg"),
]

OUTPUT_FOLDER = Path(
    "experiment_outputs/fourier_multi_image"
)

IMAGE_SIZE = (64, 64)
SEEDS = [11, 22, 33]
STEPS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 0.005
FOURIER_FEATURES = 16


# ---------------- RANDOM SEEDS ----------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------- LOAD IMAGE ----------------

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(
        IMAGE_SIZE, Image.Resampling.LANCZOS
    )

    array = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = array.shape

    y_coords, x_coords = np.meshgrid(
        np.linspace(-1, 1, height, dtype=np.float32),
        np.linspace(-1, 1, width, dtype=np.float32),
        indexing="ij"
    )

    coords = np.stack(
        [x_coords.reshape(-1), y_coords.reshape(-1)],
        axis=1
    )

    colors = array.reshape(-1, 3)

    return (
        array,
        torch.tensor(coords),
        torch.tensor(colors)
    )


# ---------------- FOURIER ENCODING ----------------

class FourierEncoding(nn.Module):
    def __init__(self, seed):
        super().__init__()

        generator = torch.Generator()
        generator.manual_seed(seed)

        matrix = torch.randn(
            2, FOURIER_FEATURES, generator=generator
        ) * 3.0

        self.register_buffer("frequency_matrix", matrix)

    def forward(self, coords):
        projected = (
            2 * np.pi * coords @ self.frequency_matrix
        )

        return torch.cat(
            [torch.sin(projected), torch.cos(projected)],
            dim=1
        )


# ---------------- MODELS ----------------

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

    def forward(self, coords):
        return self.network(coords)


class FourierMLP(nn.Module):
    def __init__(self, seed):
        super().__init__()

        self.encoding = FourierEncoding(seed)

        self.network = nn.Sequential(
            nn.Linear(FOURIER_FEATURES * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, coords):
        return self.network(self.encoding(coords))


# ---------------- ONE EXPERIMENT ----------------

def run_experiment(image_name, model_name, seed,
                   coords, colors, original):

    set_seed(seed)

    if model_name == "Baseline":
        model = BaselineMLP()
    else:
        model = FourierMLP(seed)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    loss_function = nn.MSELoss()

    start_time = time.perf_counter()
    number_of_pixels = len(coords)

    model.train()

    for step in range(STEPS):
        indices = torch.randint(
            0, number_of_pixels, (BATCH_SIZE,)
        )

        predictions = model(coords[indices])
        loss = loss_function(predictions, colors[indices])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(
                f"{image_name} | {model_name} | "
                f"Seed {seed} | Step {step + 1}/{STEPS} | "
                f"Loss {loss.item():.6f}"
            )

    model.eval()

    with torch.no_grad():
        reconstruction = model(coords).numpy()

    elapsed = time.perf_counter() - start_time

    height, width, _ = original.shape
    reconstruction = reconstruction.reshape(height, width, 3)

    mse = float(np.mean((original - reconstruction) ** 2))
    psnr = float(
        10 * np.log10(1.0 / max(mse, 1e-12))
    )

    safe_name = Path(image_name).stem
    output_path = (
        OUTPUT_FOLDER /
        f"{safe_name}_{model_name}_seed_{seed}.png"
    )

    Image.fromarray(
        np.clip(reconstruction * 255, 0, 255).astype(np.uint8)
    ).save(output_path)

    return {
        "Image": image_name,
        "Model": model_name,
        "Seed": seed,
        "MSE": mse,
        "PSNR": psnr,
        "Training_Time_sec": elapsed
    }


# ---------------- MAIN ----------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    for path in IMAGE_PATHS:
        if not path.exists():
            print(f"ERROR: Image not found: {path}")
            print("Check your test_images folder.")
            return

    results = []

    total_runs = len(IMAGE_PATHS) * 2 * len(SEEDS)
    current_run = 0

    print(f"Total runs: {total_runs}")
    print("Testing flower.jpg and place.jpg...\n")

    for image_path in IMAGE_PATHS:
        print(f"\n========== IMAGE: {image_path.name} ==========")

        original, coords, colors = load_image(image_path)

        for model_name in ["Baseline", "Fourier"]:
            for seed in SEEDS:
                current_run += 1
                print(f"\nRun {current_run}/{total_runs}")

                result = run_experiment(
                    image_path.name,
                    model_name,
                    seed,
                    coords,
                    colors,
                    original
                )

                results.append(result)

                print(
                    f"Result: MSE={result['MSE']:.6f}, "
                    f"PSNR={result['PSNR']:.2f} dB\n"
                )

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_FOLDER / "multi_image_results.csv",
        index=False
    )

    summary = (
        df.groupby(["Image", "Model"])
        .agg(
            Mean_MSE=("MSE", "mean"),
            Std_MSE=("MSE", "std"),
            Mean_PSNR=("PSNR", "mean"),
            Std_PSNR=("PSNR", "std"),
            Mean_Time_sec=("Training_Time_sec", "mean")
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_FOLDER / "multi_image_summary.csv",
        index=False
    )

    # Plot MSE for each image
    for image_name in df["Image"].unique():
        subset = df[df["Image"] == image_name]

        plt.figure(figsize=(7, 5))

        for model_name in ["Baseline", "Fourier"]:
            model_data = subset[
                subset["Model"] == model_name
            ]

            plt.scatter(
                [model_name] * len(model_data),
                model_data["MSE"],
                label=model_name
            )

        plt.ylabel("MSE (lower is better)")
        plt.title(f"Baseline vs Fourier: {image_name}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_FOLDER /
            f"{Path(image_name).stem}_mse_comparison.png",
            dpi=150
        )
        plt.close()

    print("\n========== FINAL SUMMARY ==========")
    print(summary.to_string(index=False))

    print(f"\nResults saved in: {OUTPUT_FOLDER}")
    print("Multi-image experiment completed!")


if __name__ == "__main__":
    main()
