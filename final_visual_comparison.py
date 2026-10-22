
import os
import pandas as pd
import matplotlib.pyplot as plt

# Input and output folders
INPUT_FILE = os.path.join(
    "experiment_outputs",
    "fourier_final_validation",
    "experiment_summary.csv"
)

OUTPUT_DIR = os.path.join(
    "experiment_outputs",
    "final_visual_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load experiment results
df = pd.read_csv(INPUT_FILE)

df["Error_Ratio_Percent"] = pd.to_numeric(
    df["Error_Ratio_Percent"]
)
df["Mean_MSE"] = pd.to_numeric(df["Mean_MSE"])
df["Mean_PSNR"] = pd.to_numeric(df["Mean_PSNR"])

images = df["Image"].unique()
ratios = sorted(df["Error_Ratio_Percent"].unique())

# Plot 1: MSE comparison
plt.figure(figsize=(9, 6))

for image_name in images:
    image_data = df[df["Image"] == image_name].sort_values(
        "Error_Ratio_Percent"
    )

    plt.plot(
        image_data["Error_Ratio_Percent"],
        image_data["Mean_MSE"],
        marker="o",
        linewidth=2,
        label=image_name
    )

plt.xlabel("Error-Guided Sampling Ratio (%)")
plt.ylabel("Mean MSE (lower is better)")
plt.title("Fourier Neural Painter: MSE Comparison")
plt.xticks(ratios)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

mse_path = os.path.join(OUTPUT_DIR, "mse_comparison.png")
plt.savefig(mse_path, dpi=300)
plt.close()

# Plot 2: PSNR comparison
plt.figure(figsize=(9, 6))

for image_name in images:
    image_data = df[df["Image"] == image_name].sort_values(
        "Error_Ratio_Percent"
    )

    plt.plot(
        image_data["Error_Ratio_Percent"],
        image_data["Mean_PSNR"],
        marker="o",
        linewidth=2,
        label=image_name
    )

plt.xlabel("Error-Guided Sampling Ratio (%)")
plt.ylabel("Mean PSNR in dB (higher is better)")
plt.title("Fourier Neural Painter: PSNR Comparison")
plt.xticks(ratios)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

psnr_path = os.path.join(OUTPUT_DIR, "psnr_comparison.png")
plt.savefig(psnr_path, dpi=300)
plt.close()

# Print the best ratio for each image
print("\nFINAL EXPERIMENT RESULTS")
print("=" * 55)

for image_name in images:
    image_data = df[df["Image"] == image_name]

    baseline_row = image_data[
        image_data["Error_Ratio_Percent"] == 0
    ].iloc[0]

    best_row = image_data.loc[image_data["Mean_MSE"].idxmin()]

    baseline_mse = baseline_row["Mean_MSE"]
    best_mse = best_row["Mean_MSE"]

    improvement = (
        (baseline_mse - best_mse) / baseline_mse
    ) * 100

    print(f"\nImage: {image_name}")
    print(f"Uniform MSE: {baseline_mse:.6f}")
    print(f"Best sampling ratio: {int(best_row['Error_Ratio_Percent'])}%")
    print(f"Best MSE: {best_mse:.6f}")
    print(f"MSE reduction: {improvement:.2f}%")

print("\nGraphs saved successfully:")
print(mse_path)
print(psnr_path)
print("\nExperiment comparison completed.")
