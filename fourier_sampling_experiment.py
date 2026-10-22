
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn


# =========================
# CONFIGURATION
# =========================

IMAGE_FOLDER = Path("test_images")
OUTPUT_FOLDER = Path("experiment_outputs/fourier_sampling_experiment")

IMAGE_SIZE = (64, 64)
SEEDS = [11, 22, 33]
STEPS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 0.005

NUM_FOURIER_FEATURES = 16
FOURIER_SCALE = 3.0
ERROR_UPDATE_INTERVAL = 100

STRATEGIES = [
    "Uniform",
    "Error-guided",
    "Hybrid_80_20",
]

IMAGE_NAMES = [
    "archi.jpg",
    "flower.jpg",
    "place.jpg",
]

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# LOAD IMAGE
# =========================

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    height, width, _ = image_array.shape

    y_coordinates = torch.linspace(-1, 1, height)
    x_coordinates = torch.linspace(-1, 1, width)

    yy, xx = torch.meshgrid(
        y_coordinates,
        x_coordinates,
        indexing="ij",
    )

    coordinates = torch.stack(
        [xx.reshape(-1), yy.reshape(-1)],
        dim=1,
    )

    pixels = torch.tensor(
        image_array.reshape(-1, 3),
        dtype=torch.float32,
    )

    return image_array, coordinates, pixels


# =========================
# FOURIER FEATURES
# =========================

class FourierFeatures(nn.Module):
    def __init__(self, seed):
        super().__init__()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        matrix = torch.randn(
            2,
            NUM_FOURIER_FEATURES,
            generator=generator,
        ) * FOURIER_SCALE

        self.register_buffer("matrix", matrix)

    def forward(self, coordinates):
        projected = 2 * np.pi * coordinates @ self.matrix

        return torch.cat(
            [
                torch.sin(projected),
                torch.cos(projected),
            ],
            dim=1,
        )


# =========================
# NEURAL PAINTER MODEL
# =========================

class NeuralPainter(nn.Module):
    def __init__(self, seed):
        super().__init__()

        self.fourier = FourierFeatures(seed)

        self.network = nn.Sequential(
            nn.Linear(NUM_FOURIER_FEATURES * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, coordinates):
        features = self.fourier(coordinates)
        return self.network(features)


# =========================
# SAMPLING PROBABILITIES
# =========================

def calculate_sampling_probabilities(
    strategy,
    pixel_errors,
):
    number_of_pixels = len(pixel_errors)

    uniform_probabilities = torch.full(
        (number_of_pixels,),
        1.0 / number_of_pixels,
    )

    if strategy == "Uniform":
        return uniform_probabilities

    # Avoid assigning zero probability to any pixel.
    error_weights = pixel_errors + 1e-8
    error_probabilities = error_weights / error_weights.sum()

    if strategy == "Error-guided":
        return error_probabilities

    if strategy == "Hybrid_80_20":
        return (
            0.8 * uniform_probabilities
            + 0.2 * error_probabilities
        )

    raise ValueError(f"Unknown strategy: {strategy}")


# =========================
# TRAIN ONE EXPERIMENT
# =========================

def train_model(
    image_array,
    coordinates,
    pixels,
    strategy,
    seed,
    image_name,
):
    set_seed(seed)

    model = NeuralPainter(seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    number_of_pixels = len(pixels)

    pixel_errors = torch.ones(number_of_pixels)
    sampling_probabilities = calculate_sampling_probabilities(
        strategy,
        pixel_errors,
    )

    start_time = time.perf_counter()

    for step in range(1, STEPS + 1):

        # Refresh error estimates periodically.
        if (
            strategy != "Uniform"
            and (
                step == 1
                or (step - 1) % ERROR_UPDATE_INTERVAL == 0
            )
        ):
            model.eval()

            with torch.no_grad():
                predictions = model(coordinates)
                pixel_errors = (
                    (predictions - pixels) ** 2
                ).mean(dim=1)

            sampling_probabilities = (
                calculate_sampling_probabilities(
                    strategy,
                    pixel_errors,
                )
            )

            model.train()

        # Sample training pixels.
        if strategy == "Uniform":
            indices = torch.randint(
                0,
                number_of_pixels,
                (BATCH_SIZE,),
            )
        else:
            indices = torch.multinomial(
                sampling_probabilities,
                BATCH_SIZE,
                replacement=True,
            )

        batch_coordinates = coordinates[indices]
        batch_pixels = pixels[indices]

        predictions = model(batch_coordinates)
        loss = ((predictions - batch_pixels) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(
                f"{image_name} | {strategy} | "
                f"Seed {seed} | Step {step}/{STEPS} | "
                f"Batch Loss: {loss.item():.6f}"
            )

    elapsed_time = time.perf_counter() - start_time

    # Evaluate on EVERY pixel, not just sampled pixels.
    model.eval()

    with torch.no_grad():
        reconstructed_pixels = model(coordinates)
        full_mse = (
            (reconstructed_pixels - pixels) ** 2
        ).mean().item()

        psnr = -10 * np.log10(max(full_mse, 1e-12))

    reconstructed_image = (
        reconstructed_pixels
        .reshape(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
        .cpu()
        .numpy()
    )

    reconstructed_image = np.clip(
        reconstructed_image * 255,
        0,
        255,
    ).astype(np.uint8)

    safe_image_name = Path(image_name).stem
    output_image = OUTPUT_FOLDER / (
        f"{safe_image_name}_{strategy}_seed_{seed}.png"
    )

    Image.fromarray(reconstructed_image).save(output_image)

    return {
        "Image": image_name,
        "Strategy": strategy,
        "Seed": seed,
        "MSE": full_mse,
        "PSNR": psnr,
        "Time_sec": elapsed_time,
        "Reconstruction": str(output_image),
    }


# =========================
# PLOT COMPARISONS
# =========================

def save_comparison_plots(summary):
    metrics = [
        ("Mean_MSE", "Mean Reconstruction MSE", "mse_comparison.png"),
        ("Mean_PSNR", "Mean PSNR (dB)", "psnr_comparison.png"),
    ]

    for metric, ylabel, filename in metrics:
        pivot = summary.pivot(
            index="Image",
            columns="Strategy",
            values=metric,
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(10, 5),
        )

        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Image")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Sampling Strategy")
        plt.tight_layout()

        plt.savefig(OUTPUT_FOLDER / filename, dpi=150)
        plt.close()


# =========================
# MAIN EXPERIMENT
# =========================

def main():
    print("=" * 65)
    print("FOURIER FEATURES + SAMPLING STRATEGIES")
    print("=" * 65)

    all_results = []

    for image_name in IMAGE_NAMES:
        image_path = IMAGE_FOLDER / image_name

        if not image_path.exists():
            print(
                f"\nWARNING: Image not found: {image_path}"
            )
            continue

        print(f"\nLoading image: {image_name}")

        image_array, coordinates, pixels = load_image(
            image_path
        )

        # Save the resized original for visual comparison.
        original_path = OUTPUT_FOLDER / (
            f"{Path(image_name).stem}_original.png"
        )

        Image.fromarray(
            (image_array * 255).astype(np.uint8)
        ).save(original_path)

        for strategy in STRATEGIES:
            for seed in SEEDS:
                print(
                    f"\nStarting: {image_name} | "
                    f"{strategy} | Seed {seed}"
                )

                result = train_model(
                    image_array,
                    coordinates,
                    pixels,
                    strategy,
                    seed,
                    image_name,
                )

                all_results.append(result)

                print(
                    f"Finished | MSE: {result['MSE']:.6f} | "
                    f"PSNR: {result['PSNR']:.2f} dB | "
                    f"Time: {result['Time_sec']:.2f}s"
                )

                # Save progress after every completed run.
                pd.DataFrame(all_results).to_csv(
                    OUTPUT_FOLDER / "all_results.csv",
                    index=False,
                )

    if not all_results:
        print(
            "\nNo experiments were run. "
            "Check that test_images contains the three images."
        )
        return

    results_df = pd.DataFrame(all_results)

    summary = (
        results_df
        .groupby(["Image", "Strategy"], as_index=False)
        .agg(
            Mean_MSE=("MSE", "mean"),
            Std_MSE=("MSE", "std"),
            Mean_PSNR=("PSNR", "mean"),
            Std_PSNR=("PSNR", "std"),
            Mean_Time_sec=("Time_sec", "mean"),
        )
    )

    summary["Std_MSE"] = summary["Std_MSE"].fillna(0)
    summary["Std_PSNR"] = summary["Std_PSNR"].fillna(0)

    summary.to_csv(
        OUTPUT_FOLDER / "experiment_summary.csv",
        index=False,
    )

    save_comparison_plots(summary)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(summary.to_string(index=False))

    print(f"\nAll outputs saved in: {OUTPUT_FOLDER.resolve()}")
    print("Experiment completed!")


if __name__ == "__main__":
    main()
