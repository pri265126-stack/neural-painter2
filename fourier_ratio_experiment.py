
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn


# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_FOLDER = Path("test_images")
OUTPUT_FOLDER = Path("experiment_outputs/fourier_ratio_experiment")

IMAGE_NAMES = ["archi.jpg", "flower.jpg", "place.jpg"]
SEEDS = [11, 22, 33]

IMAGE_SIZE = (64, 64)
STEPS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 0.005

NUM_FOURIER_FEATURES = 16
FOURIER_SCALE = 3.0
ERROR_UPDATE_INTERVAL = 100

# Values represent the percentage of error-guided sampling.
ERROR_RATIOS = [0.10, 0.20, 0.30]

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# REPRODUCIBILITY
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    height, width, _ = image_array.shape

    ys = torch.linspace(-1, 1, height)
    xs = torch.linspace(-1, 1, width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    coordinates = torch.stack(
        [xx.reshape(-1), yy.reshape(-1)],
        dim=1,
    )

    pixels = torch.tensor(
        image_array.reshape(-1, 3),
        dtype=torch.float32,
    )

    return image_array, coordinates, pixels


# ==========================================
# FOURIER FEATURES
# ==========================================

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
            [torch.sin(projected), torch.cos(projected)],
            dim=1,
        )


# ==========================================
# NEURAL PAINTER
# ==========================================

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


# ==========================================
# SAMPLING PROBABILITIES
# ==========================================

def get_probabilities(pixel_errors, error_ratio):
    number_of_pixels = len(pixel_errors)

    uniform = torch.full(
        (number_of_pixels,),
        1.0 / number_of_pixels,
    )

    error_weights = pixel_errors + 1e-8
    error_probabilities = error_weights / error_weights.sum()

    return (
        (1.0 - error_ratio) * uniform
        + error_ratio * error_probabilities
    )


# ==========================================
# TRAIN ONE MODEL
# ==========================================

def train_one(image_array, coordinates, pixels,
              image_name, seed, error_ratio):

    set_seed(seed)

    model = NeuralPainter(seed)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    pixel_count = len(pixels)

    pixel_errors = torch.ones(pixel_count)

    sampling_probabilities = get_probabilities(
        pixel_errors,
        error_ratio,
    )

    start_time = time.perf_counter()

    for step in range(1, STEPS + 1):

        if (
            step == 1
            or (step - 1) % ERROR_UPDATE_INTERVAL == 0
        ):
            model.eval()

            with torch.no_grad():
                predictions = model(coordinates)

                pixel_errors = (
                    (predictions - pixels) ** 2
                ).mean(dim=1)

            sampling_probabilities = get_probabilities(
                pixel_errors,
                error_ratio,
            )

            model.train()

        indices = torch.multinomial(
            sampling_probabilities,
            BATCH_SIZE,
            replacement=True,
        )

        predicted = model(coordinates[indices])
        target = pixels[indices]

        loss = ((predicted - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(
                f"{image_name} | {int(error_ratio * 100)}% "
                f"error-guided | Seed {seed} | "
                f"Step {step}/{STEPS} | "
                f"Loss: {loss.item():.6f}"
            )

    elapsed = time.perf_counter() - start_time

    model.eval()

    with torch.no_grad():
        reconstructed = model(coordinates)
        mse = ((reconstructed - pixels) ** 2).mean().item()

    psnr = -10 * np.log10(max(mse, 1e-12))

    reconstructed_array = (
        reconstructed
        .reshape(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
        .cpu()
        .numpy()
    )

    reconstructed_array = np.clip(
        reconstructed_array * 255,
        0,
        255,
    ).astype(np.uint8)

    image_stem = Path(image_name).stem
    ratio_label = int(error_ratio * 100)

    output_path = OUTPUT_FOLDER / (
        f"{image_stem}_ratio_{ratio_label}_seed_{seed}.png"
    )

    Image.fromarray(reconstructed_array).save(output_path)

    return {
        "Image": image_name,
        "Error_Ratio_Percent": ratio_label,
        "Seed": seed,
        "MSE": mse,
        "PSNR": psnr,
        "Time_sec": elapsed,
        "Reconstruction": str(output_path),
    }


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():
    print("=" * 65)
    print("FOURIER FEATURES: SAMPLING RATIO EXPERIMENT")
    print("=" * 65)

    all_results = []

    for image_name in IMAGE_NAMES:
        image_path = IMAGE_FOLDER / image_name

        if not image_path.exists():
            print(f"WARNING: Missing image: {image_path}")
            continue

        print(f"\nLoading {image_name}")

        image_array, coordinates, pixels = load_image(image_path)

        original_path = OUTPUT_FOLDER / (
            f"{Path(image_name).stem}_original.png"
        )

        Image.fromarray(
            (image_array * 255).astype(np.uint8)
        ).save(original_path)

        for ratio in ERROR_RATIOS:
            for seed in SEEDS:
                print(
                    f"\nStarting {image_name}, "
                    f"error ratio={int(ratio * 100)}%, seed={seed}"
                )

                result = train_one(
                    image_array,
                    coordinates,
                    pixels,
                    image_name,
                    seed,
                    ratio,
                )

                all_results.append(result)

                pd.DataFrame(all_results).to_csv(
                    OUTPUT_FOLDER / "all_results.csv",
                    index=False,
                )

                print(
                    f"Completed | MSE={result['MSE']:.6f} | "
                    f"PSNR={result['PSNR']:.2f} dB | "
                    f"Time={result['Time_sec']:.2f}s"
                )

    if not all_results:
        print("No runs completed. Check the test_images folder.")
        return

    results = pd.DataFrame(all_results)

    summary = (
        results
        .groupby(
            ["Image", "Error_Ratio_Percent"],
            as_index=False,
        )
        .agg(
            Mean_MSE=("MSE", "mean"),
            Std_MSE=("MSE", "std"),
            Mean_PSNR=("PSNR", "mean"),
            Std_PSNR=("PSNR", "std"),
            Mean_Time_sec=("Time_sec", "mean"),
        )
    )

    summary = summary.fillna(0)

    summary.to_csv(
        OUTPUT_FOLDER / "experiment_summary.csv",
        index=False,
    )

    # Plot MSE comparisons separately for each image.
    for image_name in results["Image"].unique():
        subset = summary[summary["Image"] == image_name]

        plt.figure(figsize=(8, 5))
        plt.errorbar(
            subset["Error_Ratio_Percent"],
            subset["Mean_MSE"],
            yerr=subset["Std_MSE"],
            marker="o",
            capsize=5,
        )

        plt.title(f"Sampling Ratio vs MSE: {image_name}")
        plt.xlabel("Error-guided sampling (%)")
        plt.ylabel("Mean MSE (lower is better)")
        plt.xticks(ERROR_RATIOS and [10, 20, 30])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_FOLDER / (
                f"{Path(image_name).stem}_ratio_mse.png"
            ),
            dpi=150,
        )
        plt.close()

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(summary.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_FOLDER.resolve()}")
    print("Sampling ratio experiment completed!")


if __name__ == "__main__":
    main()
