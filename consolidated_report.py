import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(
    "experiment_outputs",
    "consolidated_report"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# --------------------------------------------------
# 1. LOAD EXISTING EXPERIMENT RESULTS
# --------------------------------------------------

fourier = load_csv(
    "experiment_outputs/fourier_experiment/fourier_summary.csv"
)

multi = load_csv(
    "experiment_outputs/fourier_multi_image/multi_image_summary.csv"
)

sampling = load_csv(
    "experiment_outputs/fourier_final_validation/experiment_summary.csv"
)

fourier["Image"] = "archi.jpg"

fourier_all = pd.concat(
    [fourier, multi],
    ignore_index=True,
    sort=False
)

required_fourier = [
    "Model", "Image", "Mean_MSE", "Mean_PSNR",
    "Std_MSE", "Std_PSNR", "Mean_Time_sec"
]

missing = [
    col for col in required_fourier
    if col not in fourier_all.columns
]

if missing:
    raise ValueError(
        f"Missing columns in Fourier results: {missing}. "
        "Check the source CSV column names."
    )

fourier_all.to_csv(
    os.path.join(OUTPUT_DIR, "fourier_comparison.csv"),
    index=False
)

sampling.to_csv(
    os.path.join(OUTPUT_DIR, "sampling_validation.csv"),
    index=False
)


# --------------------------------------------------
# 2. BASELINE VS FOURIER COMPARISON
# --------------------------------------------------

comparison = []

for image_name in fourier_all["Image"].dropna().unique():
    data = fourier_all[
        fourier_all["Image"] == image_name
    ]

    baseline = data[
        data["Model"].astype(str).str.lower() == "baseline"
    ]

    fourier_model = data[
        data["Model"].astype(str).str.lower() == "fourier"
    ]

    if baseline.empty or fourier_model.empty:
        print(f"Skipping {image_name}: model row missing.")
        continue

    b = baseline.iloc[0]
    f = fourier_model.iloc[0]

    reduction = (
        (b["Mean_MSE"] - f["Mean_MSE"]) / b["Mean_MSE"] * 100
        if b["Mean_MSE"] != 0 else 0
    )

    comparison.append({
        "Image": image_name,
        "Baseline_MSE": b["Mean_MSE"],
        "Baseline_Std_MSE": b["Std_MSE"],
        "Baseline_PSNR": b["Mean_PSNR"],
        "Baseline_Std_PSNR": b["Std_PSNR"],
        "Baseline_Time_sec": b["Mean_Time_sec"],
        "Fourier_MSE": f["Mean_MSE"],
        "Fourier_Std_MSE": f["Std_MSE"],
        "Fourier_PSNR": f["Mean_PSNR"],
        "Fourier_Std_PSNR": f["Std_PSNR"],
        "Fourier_Time_sec": f["Mean_Time_sec"],
        "MSE_Reduction_Percent": reduction
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "fourier_mse_comparison.csv"),
    index=False
)

print("\n===== BASELINE VS FOURIER =====")
print(comparison_df.round(5).to_string(index=False))


if not comparison_df.empty:
    ax = comparison_df.set_index("Image")[
        ["Baseline_MSE", "Fourier_MSE"]
    ].plot(kind="bar", figsize=(9, 5))

    ax.set_title("Baseline MLP vs Fourier Feature MLP")
    ax.set_ylabel("Mean MSE (lower is better)")
    ax.set_xlabel("Image")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "fourier_mse_comparison.png"),
        dpi=300
    )
    plt.close()


# --------------------------------------------------
# 3. SAMPLING VALIDATION
# --------------------------------------------------

ratio_col = find_column(
    sampling,
    ["Error_Ratio_Percent", "Ratio_Percent"]
)

if ratio_col is None:
    raise ValueError(
        "Could not find the sampling ratio column in sampling CSV."
    )

for col in ["Mean_MSE", "Std_MSE", "Mean_PSNR",
            "Std_PSNR", "Mean_Time_sec", ratio_col]:
    if col in sampling.columns:
        sampling[col] = pd.to_numeric(
            sampling[col], errors="coerce"
        )

sampling = sampling.dropna(
    subset=["Image", "Mean_MSE", ratio_col]
)

sampling_results = []

for image_name in sampling["Image"].dropna().unique():
    data = sampling[
        sampling["Image"] == image_name
    ].copy()

    uniform_rows = data[data[ratio_col] == 0]

    if uniform_rows.empty:
        print(f"Skipping {image_name}: uniform result missing.")
        continue

    uniform = uniform_rows.iloc[0]
    best = data.loc[data["Mean_MSE"].idxmin()]

    uniform_mse = float(uniform["Mean_MSE"])
    best_mse = float(best["Mean_MSE"])

    reduction = (
        (uniform_mse - best_mse) / uniform_mse * 100
        if uniform_mse != 0 else 0
    )

    sampling_results.append({
        "Image": image_name,
        "Uniform_MSE": uniform_mse,
        "Best_Ratio_Percent": int(best[ratio_col]),
        "Best_MSE": best_mse,
        "MSE_Reduction_Percent": reduction
    })

    data = data.sort_values(ratio_col)

    plt.figure(figsize=(7, 5))
    plt.plot(
        data[ratio_col],
        data["Mean_MSE"],
        marker="o"
    )
    plt.title(f"Sampling Ratio Validation: {image_name}")
    plt.xlabel("Error-guided sampling ratio (%)")
    plt.ylabel("Mean MSE (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = os.path.splitext(image_name)[0]
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"sampling_{filename}.png"),
        dpi=300
    )
    plt.close()

sampling_df = pd.DataFrame(sampling_results)

sampling_df.to_csv(
    os.path.join(OUTPUT_DIR, "sampling_best_results.csv"),
    index=False
)

print("\n===== SAMPLING VALIDATION =====")
print(sampling_df.round(5).to_string(index=False))


# --------------------------------------------------
# 4. WRITE THE CONSOLIDATED TEXT REPORT
# --------------------------------------------------

lines = [
    "NEURAL PAINTER - CONSOLIDATED EXPERIMENT REPORT",
    "=" * 55,
    "",
    "1. OBJECTIVE",
    "Reconstruct an image from its pixel coordinates using an MLP.",
    "Compare a raw-coordinate Baseline MLP with a Fourier-feature MLP.",
    "Evaluate hybrid error-guided sampling at different ratios.",
    "",
    "2. EXPERIMENTAL SETUP",
    "Images: archi.jpg, flower.jpg, place.jpg.",
    "Image resolution: 64 x 64 pixels.",
    "Evaluation metrics: MSE, PSNR, and training time.",
    "Fourier comparison: results aggregated across multiple seeds.",
    "Sampling validation: 3 images x 4 ratios x 5 seeds = 60 runs.",
    "",
    "3. BASELINE VS FOURIER RESULTS",
    "MSE is better when lower; PSNR is better when higher.",
    "Standard deviations describe variation across the tested runs.",
    ""
]

for _, row in comparison_df.iterrows():
    lines.extend([
        f"Image: {row['Image']}",
        f"  Baseline mean MSE: {row['Baseline_MSE']:.6f}",
        f"  Baseline MSE standard deviation: "
        f"{row['Baseline_Std_MSE']:.6f}",
        f"  Baseline mean PSNR: {row['Baseline_PSNR']:.2f} dB",
        f"  Baseline mean training time: "
        f"{row['Baseline_Time_sec']:.2f} seconds",
        f"  Fourier mean MSE: {row['Fourier_MSE']:.6f}",
        f"  Fourier MSE standard deviation: "
        f"{row['Fourier_Std_MSE']:.6f}",
        f"  Fourier mean PSNR: {row['Fourier_PSNR']:.2f} dB",
        f"  Fourier mean training time: "
        f"{row['Fourier_Time_sec']:.2f} seconds",
        f"  MSE reduction: {row['MSE_Reduction_Percent']:.2f}%",
        ""
    ])

lines.extend([
    "4. SAMPLING VALIDATION RESULTS",
    "Each tested ratio was evaluated against uniform sampling (0%).",
    "The best ratio below means the lowest mean MSE among tested ratios.",
    ""
])

for _, row in sampling_df.iterrows():
    lines.extend([
        f"Image: {row['Image']}",
        f"  Uniform sampling MSE: {row['Uniform_MSE']:.6f}",
        f"  Best tested ratio: {row['Best_Ratio_Percent']}%",
        f"  Best MSE: {row['Best_MSE']:.6f}",
        f"  MSE reduction vs uniform: "
        f"{row['MSE_Reduction_Percent']:.2f}%",
        ""
    ])

lines.extend([
    "5. INTERPRETATION",
    "The Fourier model achieved lower mean MSE than the baseline on the",
    "images represented in the available comparison results.",
    "Sampling results varied by image; no single tested ratio was best",
    "for every image. These findings apply only to the tested settings.",
    "",
    "6. LIMITATIONS",
    "- Only three images were used for the final sampling validation.",
    "- The images were resized to 64 x 64 pixels.",
    "- Results may not generalize to other images or resolutions.",
    "- The best sampling ratio may vary by image.",
    "- Results depend on model settings, random seeds, and implementation.",
    "- Training time depends on the computer and software environment.",
    "- The experiments do not establish that Fourier features or a sampling",
    "  ratio will perform best for every possible image.",
    "",
    "7. CONCLUSION",
    "In these experiments, Fourier feature encoding substantially reduced",
    "reconstruction error relative to the baseline on the tested images.",
    "Hybrid sampling also improved mean MSE over uniform sampling for the",
    "tested images, with the best ratio differing across images.",
    "Further validation on more images and resolutions is needed.",
    "",
    "Generated from the project's saved experiment CSV files."
])

report_path = os.path.join(
    OUTPUT_DIR,
    "consolidated_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print("\nReport created successfully!")
print("Output folder:", os.path.abspath(OUTPUT_DIR))
print("Text report:", os.path.abspath(report_path))