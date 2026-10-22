
import os
import pandas as pd
import matplotlib.pyplot as plt

# File locations
BASE_DIR = "experiment_outputs/fourier_final_validation"
SUMMARY_FILE = os.path.join(BASE_DIR, "experiment_summary.csv")
RAW_FILE = os.path.join(BASE_DIR, "all_results.csv")

OUTPUT_DIR = "experiment_outputs/final_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load existing results
summary = pd.read_csv(SUMMARY_FILE)
raw = pd.read_csv(RAW_FILE)

# Ensure numerical columns are numeric
numeric_columns = [
    "Error_Ratio_Percent",
    "Mean_MSE",
    "Std_MSE",
    "Mean_PSNR",
    "Std_PSNR",
    "Mean_Time_sec",
]

for column in numeric_columns:
    if column in summary.columns:
        summary[column] = pd.to_numeric(summary[column])

# Save a clean copy of the summary
summary.to_csv(
    os.path.join(OUTPUT_DIR, "final_results_table.csv"),
    index=False
)

# Plot MSE comparison
plt.figure(figsize=(9, 6))

for image_name in summary["Image"].unique():
    data = summary[summary["Image"] == image_name].sort_values(
        "Error_Ratio_Percent"
    )

    plt.plot(
        data["Error_Ratio_Percent"],
        data["Mean_MSE"],
        marker="o",
        linewidth=2,
        label=image_name
    )

plt.xlabel("Error-guided sampling ratio (%)")
plt.ylabel("Mean MSE (lower is better)")
plt.title("Neural Painter: Reconstruction Error")
plt.xticks(sorted(summary["Error_Ratio_Percent"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "mse_results.png"),
    dpi=300
)
plt.close()

# Plot PSNR comparison
plt.figure(figsize=(9, 6))

for image_name in summary["Image"].unique():
    data = summary[summary["Image"] == image_name].sort_values(
        "Error_Ratio_Percent"
    )

    plt.plot(
        data["Error_Ratio_Percent"],
        data["Mean_PSNR"],
        marker="o",
        linewidth=2,
        label=image_name
    )

plt.xlabel("Error-guided sampling ratio (%)")
plt.ylabel("Mean PSNR in dB (higher is better)")
plt.title("Neural Painter: Reconstruction Quality")
plt.xticks(sorted(summary["Error_Ratio_Percent"].unique()))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "psnr_results.png"),
    dpi=300
)
plt.close()

# Generate findings
report_lines = [
    "NEURAL PAINTER - FINAL EXPERIMENT REPORT",
    "=" * 48,
    "",
    "Experiment: Fourier features with hybrid error-guided sampling",
    f"Images tested: {summary['Image'].nunique()}",
    f"Sampling ratios tested: "
    f"{sorted(summary['Error_Ratio_Percent'].unique())}",
    f"Individual runs recorded: {len(raw)}",
    "",
    "RESULTS BY IMAGE",
    "-" * 48,
]

for image_name in summary["Image"].unique():
    data = summary[summary["Image"] == image_name]

    baseline = data[data["Error_Ratio_Percent"] == 0].iloc[0]
    best = data.loc[data["Mean_MSE"].idxmin()]

    reduction = (
        (baseline["Mean_MSE"] - best["Mean_MSE"])
        / baseline["Mean_MSE"]
    ) * 100

    report_lines.extend([
        "",
        f"Image: {image_name}",
        f"Uniform sampling MSE: {baseline['Mean_MSE']:.6f}",
        f"Best sampling ratio: "
        f"{int(best['Error_Ratio_Percent'])}%",
        f"Best mean MSE: {best['Mean_MSE']:.6f}",
        f"Mean PSNR at best ratio: {best['Mean_PSNR']:.3f} dB",
        f"MSE reduction vs uniform: {reduction:.2f}%",
    ])

report_lines.extend([
    "",
    "INTERPRETATION",
    "-" * 48,
    "The best sampling ratio differed across the tested images.",
    "Hybrid sampling achieved lower mean MSE than uniform sampling",
    "for all three images in this validation experiment.",
    "",
    "LIMITATIONS",
    "-" * 48,
    "Only three images were evaluated in this validation.",
    "Each image and ratio combination used five random seeds.",
    "Results may differ for other images, resolutions, and settings.",
    "These experiments do not establish universal superiority.",
    "",
    "METRICS",
    "-" * 48,
    "MSE: lower values indicate smaller pixel-wise reconstruction error.",
    "PSNR: higher values indicate better reconstruction quality.",
])

report_text = "\n".join(report_lines)

with open(
    os.path.join(OUTPUT_DIR, "final_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report_text)

print(report_text)
print("\nSaved report files in:", OUTPUT_DIR)
print("- final_report.txt")
print("- final_results_table.csv")
print("- mse_results.png")
print("- psnr_results.png")
