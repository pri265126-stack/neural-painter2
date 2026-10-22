
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


# -------------------- SETTINGS --------------------

IMAGE_PATH = Path("test_images/archi.jpg")
OUTPUT_FOLDER = Path("experiment_outputs/fourier_experiment")

IMAGE_SIZE = (64, 64)
SEEDS = [11, 22, 33]
STEPS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 0.005

FOURIER_FEATURES = 16


# -------------------- REPRODUCIBILITY --------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------- LOAD IMAGE --------------------

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = image_array.shape

    y_coords, x_coords = np.meshgrid(
        np.linspace(-1, 1, height, dtype=np.float32),
        np.linspace(-1, 1, width, dtype=np.float32),
        indexing="ij"
    )

    coordinates = np.stack(
        [x_coords.reshape(-1), y_coords.reshape(-1)],
        axis=1
    )

    colors = image_array.reshape(-1, 3)

    return image_array, torch.tensor(coordinates), torch.tensor(colors)


# -------------------- FOURIER ENCODING --------------------

class FourierEncoding(nn.Module):
    def __init__(self, seed, number_of_features):
        super().__init__()

        generator = torch.Generator()
        generator.manual_seed(seed)

        matrix = torch.randn(
            2, number_of_features, generator=generator
        ) * 3.0

        self.register_buffer("frequency_matrix", matrix)

    def forward(self, coordinates):
        projected = 2 * np.pi * coordinates @ self.frequency_matrix

        return torch.cat(
            [torch.sin(projected), torch.cos(projected)],
            dim=1
        )


# -------------------- MODELS --------------------

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

    def forward(self, coordinates):
        return self.network(coordinates)


class FourierMLP(nn.Module):
    def __init__(self, seed):
        super().__init__()

        self.encoding = FourierEncoding(seed, FOURIER_FEATURES)

        self.network = nn.Sequential(
            nn.Linear(FOURIER_FEATURES * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, coordinates):
        encoded = self.encoding(coordinates)
        return self.network(encoded)


# -------------------- TRAIN AND EVALUATE --------------------

def run_experiment(model_name, seed, coordinates, colors, original_image):
    set_seed(seed)

    if model_name == "Baseline":
        model = BaselineMLP()
    else:
        model = FourierMLP(seed)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    loss_function = nn.MSELoss()

    number_of_pixels = len(coordinates)
    start_time = time.perf_counter()

    model.train()

    for step in range(STEPS):
        indices = torch.randint(
            0, number_of_pixels, (BATCH_SIZE,)
        )

        batch_coordinates = coordinates[indices]
        batch_colors = colors[indices]

        predictions = model(batch_coordinates)
        loss = loss_function(predictions, batch_colors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(
                f"{model_name} | Seed {seed} | "
                f"Step {step + 1}/{STEPS} | "
                f"Loss: {loss.item():.6f}"
            )

    model.eval()

    with torch.no_grad():
        reconstructed = model(coordinates).numpy()

    elapsed_time = time.perf_counter() - start_time

    height, width, _ = original_image.shape
    reconstructed = reconstructed.reshape(height, width, 3)
    reconstructed = np.clip(reconstructed, 0, 1)

    mse = float(np.mean((original_image - reconstructed) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_FOLDER / f"{model_name}_seed_{seed}.png"
    Image.fromarray(
        (reconstructed * 255).astype(np.uint8)
    ).save(output_path)

    return {
        "Model": model_name,
        "Seed": seed,
        "MSE": mse,
        "PSNR": psnr,
        "Training_Time_sec": elapsed_time
    }


# -------------------- MAIN --------------------

def main():
    if not IMAGE_PATH.exists():
        print(f"\nERROR: Image not found: {IMAGE_PATH}")
        print("Check that test_images/archi.jpg exists.")
        return

    print("Loading image...")
    original_image, coordinates, colors = load_image(IMAGE_PATH)

    print(f"Image size: {IMAGE_SIZE}")
    print(f"Seeds: {SEEDS}")
    print(f"Training steps per run: {STEPS}")
    print("\nStarting controlled comparison...\n")

    results = []

    for model_name in ["Baseline", "Fourier"]:
        for seed in SEEDS:
            result = run_experiment(
                model_name,
                seed,
                coordinates,
                colors,
                original_image
            )
            results.append(result)

            print(
                f"Completed: {model_name}, seed={seed}, "
                f"MSE={result['MSE']:.6f}, "
                f"PSNR={result['PSNR']:.2f} dB\n"
            )

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_FOLDER / "fourier_results.csv"
    results_df.to_csv(csv_path, index=False)

    summary = (
        results_df.groupby("Model")
        .agg(
            Mean_MSE=("MSE", "mean"),
            Std_MSE=("MSE", "std"),
            Mean_PSNR=("PSNR", "mean"),
            Std_PSNR=("PSNR", "std"),
            Mean_Time_sec=("Training_Time_sec", "mean")
        )
        .reset_index()
    )

    summary_path = OUTPUT_FOLDER / "fourier_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Comparison graph: MSE
    plt.figure(figsize=(7, 5))

    for model_name in ["Baseline", "Fourier"]:
        subset = results_df[results_df["Model"] == model_name]
        plt.scatter(
            [model_name] * len(subset),
            subset["MSE"],
            label=model_name
        )

    plt.ylabel("Mean Squared Error (lower is better)")
    plt.title("Baseline vs Fourier Features")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "mse_comparison.png", dpi=150)
    plt.close()

    # Comparison graph: PSNR
    plt.figure(figsize=(7, 5))

    for model_name in ["Baseline", "Fourier"]:
        subset = results_df[results_df["Model"] == model_name]
        plt.scatter(
            [model_name] * len(subset),
            subset["PSNR"],
            label=model_name
        )

    plt.ylabel("PSNR in dB (higher is better)")
    plt.title("Reconstruction Quality Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "psnr_comparison.png", dpi=150)
    plt.close()

    print("\n========== FINAL SUMMARY ==========")
    print(summary.to_string(index=False))

    print(f"\nDetailed results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Reconstructed images and graphs: {OUTPUT_FOLDER}")
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
