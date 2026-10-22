
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


# Experiment settings: same as the original project
SEEDS = [11, 22, 33]
STEPS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 0.005
IMAGE_SIZE = (64, 64)

IMAGE_FOLDER = Path("test_images")
OUTPUT_FOLDER = Path("experiment_outputs") / "ratio_experiment"

# Error-guided proportion in the hybrid sampling distribution
CONFIGURATIONS = {
    "Uniform_100_0": None,
    "Hybrid_90_10": 0.10,
    "Hybrid_80_20": 0.20,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NeuralPainter(nn.Module):
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


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    original = np.array(image, dtype=np.float32) / 255.0
    height, width, _ = original.shape

    ys, xs = np.meshgrid(
        np.linspace(-1, 1, height, dtype=np.float32),
        np.linspace(-1, 1, width, dtype=np.float32),
        indexing="ij"
    )

    coordinates = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    colors = original.reshape(-1, 3)

    X = torch.tensor(coordinates, dtype=torch.float32)
    Y = torch.tensor(colors, dtype=torch.float32)

    return original, X, Y, height, width


def train_model(error_ratio, seed, X, Y, original, height, width):
    set_seed(seed)

    num_pixels = len(X)
    model = NeuralPainter()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    loss_function = nn.MSELoss()

    uniform = torch.full(
        (num_pixels,),
        1.0 / num_pixels
    )
    probabilities = uniform.clone()

    start_time = time.perf_counter()

    for step in range(STEPS):

        # Recalculate sampling probabilities every 100 steps.
        # Keep the first 100 steps uniform for every configuration.
        if (
            error_ratio is not None
            and step > 0
            and step % 100 == 0
        ):
            with torch.no_grad():
                predictions = model(X)

                pixel_errors = (
                    (predictions - Y) ** 2
                ).mean(dim=1)

                error_weights = pixel_errors + 1e-8
                error_weights = (
                    error_weights / error_weights.sum()
                )

                probabilities = (
                    error_ratio * error_weights
                    + (1.0 - error_ratio) * uniform
                )

        # Uniform baseline samples pixels uniformly.
        # Hybrid configurations sample from the mixed distribution.
        if error_ratio is None:
            indices = torch.randint(
                0, num_pixels, (BATCH_SIZE,)
            )
        else:
            indices = torch.multinomial(
                probabilities,
                BATCH_SIZE,
                replacement=True
            )

        prediction = model(X[indices])
        loss = loss_function(prediction, Y[indices])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_time = time.perf_counter() - start_time

    # Evaluate every pixel, not only the sampled pixels.
    with torch.no_grad():
        reconstructed = (
            model(X).cpu().numpy().reshape(height, width, 3)
        )

    mse = float(np.mean((original - reconstructed) ** 2))
    psnr = float(
        10 * np.log10(1.0 / max(mse, 1e-12))
    )

    return reconstructed, mse, psnr, training_time


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not IMAGE_FOLDER.exists():
        print(f"ERROR: Image folder not found: {IMAGE_FOLDER}")
        return

    image_paths = sorted(
        path for path in IMAGE_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    )

    if not image_paths:
        print("ERROR: No supported images found in test_images.")
        return

    all_results = []

    total_runs = len(image_paths) * len(SEEDS) * len(CONFIGURATIONS)
    run_number = 0

    for image_path in image_paths:
        print(f"\n{'=' * 55}")
        print(f"IMAGE: {image_path.name}")
        print(f"{'=' * 55}")

        original, X, Y, height, width = load_image(image_path)

        for seed in SEEDS:
            for config_name, error_ratio in CONFIGURATIONS.items():
                run_number += 1

                print(
                    f"\nRun {run_number}/{total_runs}: "
                    f"{config_name}, seed={seed}",
                    flush=True
                )

                reconstructed, mse, psnr, seconds = train_model(
                    error_ratio,
                    seed,
                    X,
                    Y,
                    original,
                    height,
                    width
                )

                result = {
                    "Image": image_path.name,
                    "Seed": seed,
                    "Configuration": config_name,
                    "Error_Guided_Ratio": (
                        0.0 if error_ratio is None else error_ratio
                    ),
                    "MSE": mse,
                    "PSNR_dB": psnr,
                    "Training_Time_sec": seconds
                }
                all_results.append(result)

                print(
                    f"  MSE={mse:.6f} | "
                    f"PSNR={psnr:.2f} dB | "
                    f"Time={seconds:.2f}s"
                )

                # Save one reconstruction per image/configuration
                # using the first seed.
                if seed == SEEDS[0]:
                    output_image = np.clip(
                        reconstructed * 255, 0, 255
                    ).astype(np.uint8)

                    Image.fromarray(output_image).save(
                        OUTPUT_FOLDER
                        / f"{Path(image_path.name).stem}_{config_name}.png"
                    )

    # Save individual run results
    results_path = OUTPUT_FOLDER / "ratio_experiment_results.csv"

    fieldnames = list(all_results[0].keys())

    with open(
        results_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Calculate mean and standard deviation for each image/configuration
    summary = []

    for image_path in image_paths:
        for config_name in CONFIGURATIONS:
            rows = [
                row for row in all_results
                if row["Image"] == image_path.name
                and row["Configuration"] == config_name
            ]

            mse_values = np.array([r["MSE"] for r in rows])
            psnr_values = np.array([r["PSNR_dB"] for r in rows])
            times = np.array([r["Training_Time_sec"] for r in rows])

            summary.append({
                "Image": image_path.name,
                "Configuration": config_name,
                "Mean_MSE": float(np.mean(mse_values)),
                "Std_MSE": float(np.std(mse_values)),
                "Mean_PSNR_dB": float(np.mean(psnr_values)),
                "Std_PSNR_dB": float(np.std(psnr_values)),
                "Mean_Time_sec": float(np.mean(times))
            })

    summary_path = OUTPUT_FOLDER / "ratio_experiment_summary.csv"

    with open(
        summary_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    # Create MSE and PSNR comparison charts for each image
    for image_path in image_paths:
        rows = [
            row for row in summary
            if row["Image"] == image_path.name
        ]

        names = [row["Configuration"] for row in rows]
        mse_means = [row["Mean_MSE"] for row in rows]
        mse_stds = [row["Std_MSE"] for row in rows]
        psnr_means = [row["Mean_PSNR_dB"] for row in rows]
        psnr_stds = [row["Std_PSNR_dB"] for row in rows]

        image_stem = Path(image_path.name).stem

        plt.figure(figsize=(9, 5))
        plt.bar(names, mse_means, yerr=mse_stds, capsize=5)
        plt.ylabel("Mean MSE (lower is better)")
        plt.title(f"Sampling Ratio MSE: {image_path.name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_FOLDER / f"{image_stem}_ratio_mse.png")
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.bar(names, psnr_means, yerr=psnr_stds, capsize=5)
        plt.ylabel("Mean PSNR in dB (higher is better)")
        plt.title(f"Sampling Ratio PSNR: {image_path.name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_FOLDER / f"{image_stem}_ratio_psnr.png")
        plt.close()

    print(f"\n{'=' * 60}")
    print("RATIO EXPERIMENT COMPLETED")
    print(f"Total training runs: {len(all_results)}")
    print(f"Individual results: {results_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Outputs folder: {OUTPUT_FOLDER.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
