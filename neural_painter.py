
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image



SEEDS = [11, 22, 33, 44, 55]
STEPS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 0.005
IMAGE_SIZE = (64, 64)

IMAGE_FOLDER = Path("test_images")
OUTPUT_FOLDER = Path("experiment_outputs")

STRATEGIES = ["Uniform", "Error-guided", "Hybrid"]


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

    coordinates = np.stack(
        [xs, ys], axis=-1
    ).reshape(-1, 2)

    colors = original.reshape(-1, 3)

    X = torch.tensor(coordinates)
    Y = torch.tensor(colors)

    return original, X, Y, height, width



def train_model(strategy, seed, X, Y, original, height, width):
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

        # Recalculate pixel errors every 100 steps
        if strategy != "Uniform" and step > 0 and step % 100 == 0:
            with torch.no_grad():
                predictions = model(X)

                pixel_errors = (
                    (predictions - Y) ** 2
                ).mean(dim=1)

                error_weights = pixel_errors + 1e-8
                error_weights = (
                    error_weights / error_weights.sum()
                )

                if strategy == "Error-guided":
                    probabilities = error_weights

                elif strategy == "Hybrid":
                    probabilities = (
                        0.2 * error_weights
                        + 0.8 * uniform
                    )

        if strategy == "Uniform":
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

    # Evaluate on every pixel, not just sampled pixels
    with torch.no_grad():
        reconstructed = (
            model(X).numpy().reshape(height, width, 3)
        )

    mse = float(np.mean((original - reconstructed) ** 2))
    psnr = float(
        10 * np.log10(1.0 / max(mse, 1e-12))
    )

    return reconstructed, mse, psnr, training_time




def save_visuals(image_name, strategy, seed, original, reconstructed):
    safe_name = Path(image_name).stem
    strategy_name = strategy.lower().replace("-", "_")

    # Save reconstruction
    reconstructed_image = np.clip(
        reconstructed * 255, 0, 255
    ).astype(np.uint8)

    Image.fromarray(reconstructed_image).save(
        OUTPUT_FOLDER
        / f"{safe_name}_{strategy_name}_seed{seed}.png"
    )

    # Save reconstruction error heatmap
    error_map = np.mean(
        (original - reconstructed) ** 2,
        axis=2
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(error_map, cmap="hot")
    plt.colorbar(label="Pixel MSE")
    plt.title(
        f"Error Heatmap: {safe_name} | {strategy} | Seed {seed}"
    )
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER
        / f"{safe_name}_{strategy_name}_seed{seed}_heatmap.png"
    )
    plt.close()




def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in IMAGE_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    )

    if not image_paths:
        print("ERROR: No supported images found in test_images.")
        print("Supported formats: JPG, JPEG, PNG, WEBP")
        return

    if len(image_paths) != 3:
        print(
            f"WARNING: Found {len(image_paths)} images, "
            "not 3. Continuing with all supported images."
        )

    all_results = []

    for image_path in image_paths:
        print("\n" + "=" * 55)
        print(f"IMAGE: {image_path.name}")
        print("=" * 55)

        original, X, Y, height, width = load_image(image_path)

        for seed in SEEDS:
            print(f"\nSeed: {seed}")

            for strategy in STRATEGIES:
                print(f"  Training {strategy}...", flush=True)

                reconstructed, mse, psnr, seconds = train_model(
                    strategy,
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
                    "Strategy": strategy,
                    "MSE": mse,
                    "PSNR_dB": psnr,
                    "Training_Time_sec": seconds
                }

                all_results.append(result)

                print(
                    f"    MSE={mse:.6f} | "
                    f"PSNR={psnr:.2f} dB | "
                    f"Time={seconds:.2f}s"
                )

                # Save one set of visuals per image and strategy
                if seed == SEEDS[0]:
                    save_visuals(
                        image_path.name,
                        strategy,
                        seed,
                        original,
                        reconstructed
                    )

    # Save results from all 45 runs (if there are 3 images)
    results_file = OUTPUT_FOLDER / "all_experiment_results.csv"

    with open(
        results_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Image",
                "Seed",
                "Strategy",
                "MSE",
                "PSNR_dB",
                "Training_Time_sec"
            ]
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Calculate summary separately for each image and strategy
    summary = []

    print("\n\n" + "=" * 65)
    print("FINAL SUMMARY: EACH IMAGE AND STRATEGY")
    print("=" * 65)

    for image_path in image_paths:
        for strategy in STRATEGIES:
            rows = [
                row for row in all_results
                if row["Image"] == image_path.name
                and row["Strategy"] == strategy
            ]

            mse_values = np.array([r["MSE"] for r in rows])
            psnr_values = np.array([r["PSNR_dB"] for r in rows])
            time_values = np.array(
                [r["Training_Time_sec"] for r in rows]
            )

            summary_row = {
                "Image": image_path.name,
                "Strategy": strategy,
                "Mean_MSE": float(np.mean(mse_values)),
                "Std_MSE": float(np.std(mse_values)),
                "Mean_PSNR_dB": float(np.mean(psnr_values)),
                "Std_PSNR_dB": float(np.std(psnr_values)),
                "Mean_Time_sec": float(np.mean(time_values))
            }

            summary.append(summary_row)

            print(
                f"\n{image_path.name} | {strategy}\n"
                f"  Mean MSE: {summary_row['Mean_MSE']:.6f}\n"
                f"  MSE Std:  {summary_row['Std_MSE']:.6f}\n"
                f"  Mean PSNR: {summary_row['Mean_PSNR_dB']:.2f} dB\n"
                f"  Mean Time: {summary_row['Mean_Time_sec']:.2f}s"
            )

    summary_file = OUTPUT_FOLDER / "experiment_summary.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Image",
                "Strategy",
                "Mean_MSE",
                "Std_MSE",
                "Mean_PSNR_dB",
                "Std_PSNR_dB",
                "Mean_Time_sec"
            ]
        )
        writer.writeheader()
        writer.writerows(summary)

    # Create separate comparison charts for each image
    for image_path in image_paths:
        image_summary = [
            row for row in summary
            if row["Image"] == image_path.name
        ]

        names = [row["Strategy"] for row in image_summary]
        mse_means = [row["Mean_MSE"] for row in image_summary]
        mse_stds = [row["Std_MSE"] for row in image_summary]
        psnr_means = [row["Mean_PSNR_dB"] for row in image_summary]
        psnr_stds = [row["Std_PSNR_dB"] for row in image_summary]

        safe_name = Path(image_path.name).stem

        plt.figure(figsize=(8, 5))
        plt.bar(names, mse_means, yerr=mse_stds, capsize=5)
        plt.ylabel("Mean MSE (lower is better)")
        plt.title(f"MSE Comparison: {image_path.name}")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_FOLDER / f"{safe_name}_mse_comparison.png"
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(names, psnr_means, yerr=psnr_stds, capsize=5)
        plt.ylabel("Mean PSNR in dB (higher is better)")
        plt.title(f"PSNR Comparison: {image_path.name}")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_FOLDER / f"{safe_name}_psnr_comparison.png"
        )
        plt.close()

    print("\n" + "=" * 65)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Total training runs: {len(all_results)}")
    print(f"Results saved in: {OUTPUT_FOLDER.resolve()}")
    print("Main results: all_experiment_results.csv")
    print("Summary: experiment_summary.csv")
    print("Reconstruction images, heatmaps and charts are also saved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
