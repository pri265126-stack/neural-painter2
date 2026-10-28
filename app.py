
import io
import csv
import time
from datetime import datetime

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# ================= SETTINGS =================
SIZE = 64
STEPS = 1000
CORRECTION_STEPS = 500
BATCH_SIZE = 512
LEARNING_RATE = 0.005
FOURIER_FEATURES = 16
SEED = 11
ERROR_REFRESH_INTERVAL = 50
UNIFORM_MIX = 0.5

st.set_page_config(
    page_title="Self-Correcting Neural Painter",
    page_icon="🎨",
    layout="wide",
)


# ================= MODELS =================
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, coords):
        return self.network(coords)


class FourierEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(SEED)

        matrix = torch.randn(
            2, FOURIER_FEATURES, generator=generator
        ) * 3.0

        self.register_buffer("matrix", matrix)

    def forward(self, coords):
        projected = 2 * np.pi * (coords @ self.matrix)

        return torch.cat(
            [torch.sin(projected), torch.cos(projected)],
            dim=-1,
        )


class FourierMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding = FourierEncoding()

        self.network = nn.Sequential(
            nn.Linear(FOURIER_FEATURES * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, coords):
        # FIX: 2 coordinate values become 32 Fourier features.
        encoded_coords = self.encoding(coords)
        return self.network(encoded_coords)


# ================= IMAGE PROCESSING =================
def prepare_image(uploaded_file):
    original = Image.open(uploaded_file).convert("RGB")

    small = original.resize(
        (SIZE, SIZE), Image.Resampling.LANCZOS
    )

    pixels = np.asarray(
        small, dtype=np.float32
    ) / 255.0

    axis = np.linspace(-1, 1, SIZE, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")

    coords = np.stack(
        [x, y], axis=-1
    ).reshape(-1, 2)

    colors = pixels.reshape(-1, 3)

    return (
        original,
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(colors, dtype=torch.float32),
    )


def calculate_metrics(prediction, target):
    mse = float(np.mean((prediction - target) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    return mse, psnr


def prediction_to_image(prediction):
    array = np.clip(
        prediction.reshape(SIZE, SIZE, 3), 0, 1
    )
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


# ================= TRAINING =================
def train_model(
    model, coords, colors, steps,
    progress, start_fraction, end_fraction, name
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    count = len(coords)
    batch_size = min(BATCH_SIZE, count)

    model.train()

    for step in range(steps):
        indices = torch.randint(
            0, count, (batch_size,)
        )

        optimizer.zero_grad()

        output = model(coords[indices])
        loss = torch.mean(
            (output - colors[indices]) ** 2
        )

        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0:
            fraction = start_fraction + (
                end_fraction - start_fraction
            ) * (step + 1) / steps

            progress.progress(
                min(fraction, 1.0),
                text=f"{name}: {step + 1}/{steps}",
            )

    model.eval()

    with torch.no_grad():
        return model(coords).cpu().numpy()


# ================= ADAPTIVE SELF-CORRECTION =================
def self_correct_model(model, coords, colors, progress):
    count = len(coords)
    batch_size = min(BATCH_SIZE, count)

    uniform_probs = np.full(
        count, 1.0 / count, dtype=np.float64
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE * 0.5,
    )

    def calculate_sampling_probabilities():
        model.eval()

        with torch.no_grad():
            prediction = model(coords)

            pixel_errors = torch.mean(
                (prediction - colors) ** 2,
                dim=1,
            ).cpu().numpy()

        error_probs = (
            pixel_errors.astype(np.float64) + 1e-8
        )
        error_probs /= error_probs.sum()

        # Mix uniform sampling with error-guided sampling.
        probabilities = (
            UNIFORM_MIX * uniform_probs
            + (1.0 - UNIFORM_MIX) * error_probs
        )

        probabilities /= probabilities.sum()
        return probabilities

    probabilities = calculate_sampling_probabilities()
    model.train()

    for step in range(CORRECTION_STEPS):

        # Refresh error estimates periodically.
        if (
            step > 0
            and step % ERROR_REFRESH_INTERVAL == 0
        ):
            probabilities = calculate_sampling_probabilities()
            model.train()

        selected = np.random.choice(
            count,
            size=batch_size,
            replace=True,
            p=probabilities,
        )

        indices = torch.tensor(
            selected, dtype=torch.long
        )

        optimizer.zero_grad()

        output = model(coords[indices])
        loss = torch.mean(
            (output - colors[indices]) ** 2
        )

        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 50 == 0:
            progress.progress(
                0.66 + 0.34 * (step + 1) / CORRECTION_STEPS,
                text=(
                    "Adaptive self-correction: "
                    f"{step + 1}/{CORRECTION_STEPS}"
                ),
            )

    model.eval()

    with torch.no_grad():
        return model(coords).cpu().numpy()


# ================= WEB APP =================
st.title("🎨 Self-Correcting Neural Painter")

st.write(
    "Reconstruct images using neural networks, compare "
    "Baseline MLP and Fourier MLP, and inspect adaptive "
    "error-guided sampling."
)

st.info(
    "This experiment trains on the uploaded image itself. "
    "Its metrics measure reconstruction quality, not "
    "generalization to completely unseen images."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")

else:
    try:
        original, coords, colors = prepare_image(uploaded_file)

        st.subheader("Original Image")
        st.image(original, use_container_width=True)

        st.caption(
            f"Original dimensions: {original.width} × "
            f"{original.height}. Training resolution: "
            f"{SIZE} × {SIZE}."
        )

        if st.button(
            "Train & Reconstruct",
            type="primary",
        ):
            progress = st.progress(
                0, text="Preparing models..."
            )
            status = st.empty()

            try:
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                target = colors.numpy()

                # ----- Baseline MLP -----
                status.write("Training Baseline MLP...")
                start = time.time()

                baseline_model = BaselineMLP()

                baseline_prediction = train_model(
                    baseline_model,
                    coords,
                    colors,
                    STEPS,
                    progress,
                    0.0,
                    0.33,
                    "Baseline MLP",
                )

                baseline_time = time.time() - start

                bmse, bpsnr = calculate_metrics(
                    baseline_prediction, target
                )

                # ----- Fourier MLP -----
                status.write("Training Fourier MLP...")
                start = time.time()

                fourier_model = FourierMLP()

                fourier_prediction = train_model(
                    fourier_model,
                    coords,
                    colors,
                    STEPS,
                    progress,
                    0.33,
                    0.66,
                    "Fourier MLP",
                )

                fourier_time = time.time() - start

                fmse, fpsnr = calculate_metrics(
                    fourier_prediction, target
                )

                before_mse = fmse
                before_psnr = fpsnr

                # ----- Adaptive correction -----
                status.write(
                    "Running adaptive self-correction..."
                )
                start = time.time()

                corrected_prediction = self_correct_model(
                    fourier_model,
                    coords,
                    colors,
                    progress,
                )

                correction_time = time.time() - start

                after_mse, after_psnr = calculate_metrics(
                    corrected_prediction, target
                )

                # ----- Error heatmap -----
                target_img = target.reshape(
                    SIZE, SIZE, 3
                )

                corrected_img = corrected_prediction.reshape(
                    SIZE, SIZE, 3
                )

                errors = np.sqrt(
                    np.mean(
                        (target_img - corrected_img) ** 2,
                        axis=2,
                    )
                )

                errors /= max(
                    float(errors.max()), 1e-8
                )

                heat = np.zeros(
                    (SIZE, SIZE, 3), dtype=np.uint8
                )

                heat[:, :, 0] = (
                    errors * 255
                ).astype(np.uint8)

                heat[:, :, 1] = (
                    (1 - errors) * 70
                ).astype(np.uint8)

                heat_image = Image.fromarray(
                    heat, "RGB"
                )

                improvement = (
                    (before_mse - after_mse)
                    / max(before_mse, 1e-12)
                ) * 100

                # Store outputs so Streamlit reruns don't lose them.
                st.session_state["results"] = {
                    "baseline_image": prediction_to_image(
                        baseline_prediction
                    ),
                    "fourier_image": prediction_to_image(
                        fourier_prediction
                    ),
                    "corrected_image": prediction_to_image(
                        corrected_prediction
                    ),
                    "heatmap": heat_image,
                    "bmse": bmse,
                    "bpsnr": bpsnr,
                    "baseline_time": baseline_time,
                    "fmse": fmse,
                    "fpsnr": fpsnr,
                    "fourier_time": fourier_time,
                    "before_mse": before_mse,
                    "before_psnr": before_psnr,
                    "after_mse": after_mse,
                    "after_psnr": after_psnr,
                    "correction_time": correction_time,
                    "improvement": improvement,
                    "image_name": uploaded_file.name,
                }

                progress.progress(
                    1.0, text="Training complete"
                )
                status.success(
                    "Training completed successfully!"
                )

            except Exception as error:
                status.error(
                    f"Training failed: {error}"
                )

        results = st.session_state.get("results")

        if results:
            st.divider()

            tab1, tab2, tab3, tab4 = st.tabs([
                "Model Comparison",
                "Error Heatmap",
                "Self-Correction",
                "Adaptive Sampling",
            ])

            # ----- Comparison -----
            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Baseline MLP")
                    st.image(
                        results["baseline_image"],
                        use_container_width=True,
                    )
                    st.metric(
                        "MSE",
                        f'{results["bmse"]:.6f}',
                    )
                    st.metric(
                        "PSNR",
                        f'{results["bpsnr"]:.2f} dB',
                    )
                    st.write(
                        "Training time: "
                        f'{results["baseline_time"]:.2f} sec'
                    )

                with col2:
                    st.subheader("Fourier MLP")
                    st.image(
                        results["fourier_image"],
                        use_container_width=True,
                    )
                    st.metric(
                        "MSE",
                        f'{results["fmse"]:.6f}',
                    )
                    st.metric(
                        "PSNR",
                        f'{results["fpsnr"]:.2f} dB',
                    )
                    st.write(
                        "Training time: "
                        f'{results["fourier_time"]:.2f} sec'
                    )

            # ----- Heatmap -----
            with tab2:
                st.subheader("Reconstruction Error Heatmap")

                st.image(
                    results["heatmap"],
                    use_container_width=True,
                )

                st.write(
                    "Redder areas represent higher pixel "
                    "reconstruction errors. The heatmap is "
                    "normalized for visualization."
                )

            # ----- Before vs after correction -----
            with tab3:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Before Correction")
                    st.image(
                        results["fourier_image"],
                        use_container_width=True,
                    )
                    st.metric(
                        "MSE",
                        f'{results["before_mse"]:.6f}',
                    )
                    st.metric(
                        "PSNR",
                        f'{results["before_psnr"]:.2f} dB',
                    )

                with col2:
                    st.subheader("After Correction")
                    st.image(
                        results["corrected_image"],
                        use_container_width=True,
                    )
                    st.metric(
                        "MSE",
                        f'{results["after_mse"]:.6f}',
                    )
                    st.metric(
                        "PSNR",
                        f'{results["after_psnr"]:.2f} dB',
                    )

                st.metric(
                    "MSE Improvement",
                    f'{results["improvement"]:.2f}%',
                    help=(
                        "Positive means MSE decreased; "
                        "negative means MSE increased."
                    ),
                )

                st.write(
                    "Correction time: "
                    f'{results["correction_time"]:.2f} sec'
                )

                st.caption(
                    "Adaptive sampling does not guarantee "
                    "improvement on every image or run."
                )

            # ----- Adaptive sampling explanation -----
            with tab4:
                st.markdown("""
                **How adaptive sampling works**

                1. The Fourier model predicts RGB values for each pixel.
                2. Reconstruction errors are calculated for every pixel.
                3. Pixels with larger errors get a higher selection probability.
                4. Sampling probabilities refresh every 50 correction steps.
                5. Sampling mixes 50% uniform and 50% error-guided sampling.

                **Evaluation**

                - Lower MSE indicates lower average squared reconstruction error.
                - Higher PSNR generally indicates better reconstruction quality.
                """)

            # ----- Download experiment results -----
            row = {
                "Date_Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Image_Name": results["image_name"],
                "Baseline_MSE": results["bmse"],
                "Baseline_PSNR_dB": results["bpsnr"],
                "Baseline_Time_sec": results["baseline_time"],
                "Fourier_MSE": results["fmse"],
                "Fourier_PSNR_dB": results["fpsnr"],
                "Fourier_Time_sec": results["fourier_time"],
                "Before_Correction_MSE": results["before_mse"],
                "Before_Correction_PSNR_dB": results["before_psnr"],
                "After_Correction_MSE": results["after_mse"],
                "After_Correction_PSNR_dB": results["after_psnr"],
                "Correction_Time_sec": results["correction_time"],
                "MSE_Improvement_Percent": results["improvement"],
                "Error_Refresh_Interval": ERROR_REFRESH_INTERVAL,
                "Uniform_Mix": UNIFORM_MIX,
            }

            csv_buffer = io.StringIO()

            writer = csv.DictWriter(
                csv_buffer,
                fieldnames=list(row.keys()),
            )

            writer.writeheader()
            writer.writerow(row)

            st.download_button(
                "Download Experiment Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="experiment_results_adaptive.csv",
                mime="text/csv",
            )

    except Exception as error:
        st.error(
            f"Could not process this image: {error}"
        )
