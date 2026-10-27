import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import time
import threading
import os
import csv
from datetime import datetime

# =============== SETTINGS ===============
SIZE = 64
STEPS = 1000
CORRECTION_STEPS = 500
BATCH_SIZE = 512
LEARNING_RATE = 0.005
FOURIER_FEATURES = 16
SEED = 11

ERROR_REFRESH_INTERVAL = 50
UNIFORM_MIX = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============== BASELINE MODEL ===============
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


# =============== FOURIER ENCODING ===============
class FourierEncoding(nn.Module):
    def __init__(self):
        super().__init__()

        generator = torch.Generator().manual_seed(SEED)

        matrix = torch.randn(
            2, FOURIER_FEATURES, generator=generator
        ) * 3.0

        self.register_buffer("matrix", matrix)

    def forward(self, coords):
        projected = 2 * np.pi * coords @ self.matrix

        return torch.cat(
            [torch.sin(projected), torch.cos(projected)],
            dim=-1
        )


# =============== FOURIER MODEL ===============
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
            nn.Sigmoid()
        )

    def forward(self, coords):
        return self.network(self.encoding(coords))


# =============== IMAGE PROCESSING ===============
def prepare_image(path):
    original = Image.open(path).convert("RGB")

    small = original.resize(
        (SIZE, SIZE), Image.Resampling.LANCZOS
    )

    pixels = np.asarray(small, dtype=np.float32) / 255.0

    axis = np.linspace(-1, 1, SIZE, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")

    coords = np.stack([x, y], axis=-1).reshape(-1, 2)

    return (
        original,
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(
            pixels.reshape(-1, 3), dtype=torch.float32
        )
    )


def calculate_metrics(prediction, target):
    mse = float(np.mean((prediction - target) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    return mse, psnr


def prediction_to_image(prediction):
    array = np.clip(
        prediction.reshape(SIZE, SIZE, 3), 0, 1
    )
    return Image.fromarray(
        (array * 255).astype(np.uint8)
    )


# =============== SAVE EXPERIMENT RESULTS ===============
def save_experiment_result(
    image_name,
    bmse, bpsnr, btime,
    fmse, fpsnr, ftime,
    before_mse, before_psnr,
    after_mse, after_psnr, correction_time
):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_results_adaptive.csv"
    )

    columns = [
        "Date_Time", "Image_Name",
        "Baseline_MSE", "Baseline_PSNR_dB",
        "Baseline_Time_sec",
        "Fourier_MSE", "Fourier_PSNR_dB",
        "Fourier_Time_sec",
        "Before_Correction_MSE",
        "Before_Correction_PSNR_dB",
        "After_Correction_MSE",
        "After_Correction_PSNR_dB",
        "Correction_Time_sec",
        "MSE_Improvement_Percent",
        "Error_Refresh_Interval", "Uniform_Mix"
    ]

    improvement = (
        (before_mse - after_mse) / max(before_mse, 1e-12)
    ) * 100

    row = {
        "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image_Name": image_name,
        "Baseline_MSE": round(bmse, 6),
        "Baseline_PSNR_dB": round(bpsnr, 2),
        "Baseline_Time_sec": round(btime, 2),
        "Fourier_MSE": round(fmse, 6),
        "Fourier_PSNR_dB": round(fpsnr, 2),
        "Fourier_Time_sec": round(ftime, 2),
        "Before_Correction_MSE": round(before_mse, 6),
        "Before_Correction_PSNR_dB": round(before_psnr, 2),
        "After_Correction_MSE": round(after_mse, 6),
        "After_Correction_PSNR_dB": round(after_psnr, 2),
        "Correction_Time_sec": round(correction_time, 2),
        "MSE_Improvement_Percent": round(improvement, 2),
        "Error_Refresh_Interval": ERROR_REFRESH_INTERVAL,
        "Uniform_Mix": UNIFORM_MIX
    }

    exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        if not exists or os.path.getsize(filename) == 0:
            writer.writeheader()

        writer.writerow(row)

    return filename, improvement


# =============== TRAIN MODEL ===============
def train_model(model, coords, colors, steps, status, name):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    count = len(coords)
    model.train()

    for step in range(steps):
        indices = torch.randint(
            0, count, (min(BATCH_SIZE, count),)
        )

        optimizer.zero_grad()

        output = model(coords[indices])
        loss = torch.mean((output - colors[indices]) ** 2)

        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0:
            status(f"Training {name}: {step + 1}/{steps}")

    model.eval()

    with torch.no_grad():
        return model(coords).cpu().numpy()


# =============== ADAPTIVE SELF-CORRECTION ===============
def self_correct_model(model, coords, colors, status):
    count = len(coords)
    batch_size = min(BATCH_SIZE, count)

    uniform_probs = np.full(
        count, 1.0 / count, dtype=np.float64
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE * 0.5
    )

    def get_sampling_probabilities():
        model.eval()

        with torch.no_grad():
            prediction = model(coords)

            errors = torch.mean(
                (prediction - colors) ** 2, dim=1
            ).cpu().numpy()

        error_probs = errors.astype(np.float64) + 1e-8
        error_probs /= error_probs.sum()

        probabilities = (
            UNIFORM_MIX * uniform_probs
            + (1.0 - UNIFORM_MIX) * error_probs
        )

        probabilities /= probabilities.sum()

        return probabilities

    probabilities = get_sampling_probabilities()
    model.train()

    for step in range(CORRECTION_STEPS):

        # Recalculate errors after every 50 steps.
        if step > 0 and step % ERROR_REFRESH_INTERVAL == 0:
            probabilities = get_sampling_probabilities()
            model.train()

        selected = np.random.choice(
            count,
            size=batch_size,
            replace=True,
            p=probabilities
        )

        indices = torch.tensor(selected, dtype=torch.long)

        optimizer.zero_grad()

        output = model(coords[indices])
        loss = torch.mean((output - colors[indices]) ** 2)

        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0:
            status(
                f"Adaptive self-correction: "
                f"{step + 1}/{CORRECTION_STEPS}"
            )

    model.eval()

    with torch.no_grad():
        return model(coords).cpu().numpy()


# =============== GUI APPLICATION ===============
class NeuralPainterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Self-Correcting Neural Painter")
        self.root.geometry("1450x900")
        self.root.minsize(1000, 700)

        self.original = None
        self.coords = None
        self.colors = None
        self.image_path = None
        self.photos = {}
        self.training = False

        self.build_gui()

    def build_gui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")

        ttk.Button(
            toolbar, text="Choose Image",
            command=self.choose_image
        ).pack(side="left", padx=5)

        self.file_label = ttk.Label(
            toolbar, text="No image selected"
        )
        self.file_label.pack(side="left", padx=10)

        self.train_button = ttk.Button(
            toolbar, text="Train & Reconstruct",
            command=self.start_training
        )
        self.train_button.pack(side="left", padx=5)

        self.status_text = tk.StringVar(
            value="Choose an image to begin."
        )

        ttk.Label(
            self.root, textvariable=self.status_text,
            anchor="center", padding=5
        ).pack(fill="x")

        self.progress = ttk.Progressbar(
            self.root, maximum=100, mode="determinate"
        )
        self.progress.pack(fill="x", padx=12, pady=5)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)

        self.original_tab = ttk.Frame(self.tabs)
        self.comparison_tab = ttk.Frame(self.tabs)
        self.heatmap_tab = ttk.Frame(self.tabs)
        self.correction_tab = ttk.Frame(self.tabs)
        self.info_tab = ttk.Frame(self.tabs)

        self.tabs.add(self.original_tab, text="Original")
        self.tabs.add(self.comparison_tab, text="Model Comparison")
        self.tabs.add(self.heatmap_tab, text="Error Heatmap")
        self.tabs.add(self.correction_tab, text="Adaptive Self-Correction")
        self.tabs.add(self.info_tab, text="Adaptive Sampling Info")

        self.original_view = self.make_panel(
            self.original_tab, "Original Image"
        )

        comparison = ttk.Frame(self.comparison_tab)
        comparison.pack(fill="both", expand=True)

        self.baseline_view = self.make_panel(
            comparison, "Baseline MLP", side="left"
        )
        self.fourier_view = self.make_panel(
            comparison, "Fourier MLP", side="left"
        )

        self.metrics = tk.StringVar(value="Metrics will appear here.")

        ttk.Label(
            self.comparison_tab,
            textvariable=self.metrics,
            font=("Consolas", 11),
            justify="center"
        ).pack(side="bottom", fill="x", pady=8)

        self.heatmap_view = self.make_panel(
            self.heatmap_tab, "Error Heatmap"
        )

        correction = ttk.Frame(self.correction_tab)
        correction.pack(fill="both", expand=True)

        self.before_view = self.make_panel(
            correction, "Before Adaptive Correction", side="left"
        )
        self.after_view = self.make_panel(
            correction, "After Adaptive Correction", side="left"
        )

        self.correction_metrics = tk.StringVar(value="")

        ttk.Label(
            self.correction_tab,
            textvariable=self.correction_metrics,
            font=("Consolas", 11),
            justify="center",
            padding=8
        ).pack(side="bottom", fill="x")

        info = (
            "ADAPTIVE SAMPLING\n\n"
            "1. The model predicts RGB values for each pixel.\n"
            "2. Current pixel errors are calculated.\n"
            "3. High-error pixels receive a higher chance of selection.\n"
            f"4. Sampling probabilities refresh every "
            f"{ERROR_REFRESH_INTERVAL} steps.\n\n"
            f"Uniform sampling mix: {UNIFORM_MIX:.0%}\n\n"
            "MSE lower is better. PSNR higher is generally better.\n"
            "Adaptive sampling does not guarantee improvement."
        )

        ttk.Label(
            self.info_tab, text=info,
            justify="left", font=("Segoe UI", 13), padding=24
        ).pack(anchor="nw", fill="both", expand=True)

    def make_panel(self, parent, title, side="top"):
        frame = ttk.LabelFrame(parent, text=title, padding=8)

        frame.pack(
            side=side, fill="both", expand=True,
            padx=6, pady=6
        )

        label = ttk.Label(
            frame, text="Image will appear here",
            anchor="center"
        )
        label.pack(fill="both", expand=True)

        return label

    def display_image(
        self, label, image, key,
        max_width=650, max_height=500
    ):
        image = image.copy().convert("RGB")

        width, height = image.size
        scale = min(max_width / width, max_height / height)

        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale))
        )

        image = image.resize(
            new_size, Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(image)

        label.configure(image=photo, text="")
        label.image = photo
        self.photos[key] = photo

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            self.original, self.coords, self.colors = prepare_image(path)
            self.image_path = path

            self.file_label.config(
                text=f"{os.path.basename(path)} | "
                     f"{self.original.width} x {self.original.height}"
            )

            self.display_image(
                self.original_view, self.original,
                "original", 1000, 650
            )

            self.status_text.set(
                "Image loaded. Click Train & Reconstruct."
            )
            self.progress["value"] = 0

        except Exception as error:
            messagebox.showerror("Image Error", str(error))

    def set_status(self, text):
        def update():
            self.status_text.set(text)

            try:
                current, total = map(
                    int, text.split(":")[-1].strip().split("/")
                )
            except ValueError:
                return

            if "Baseline MLP:" in text:
                self.progress["value"] = current / total * 33
            elif "Fourier MLP:" in text:
                self.progress["value"] = 33 + current / total * 33
            elif "Adaptive self-correction:" in text:
                self.progress["value"] = 66 + current / total * 34

        self.root.after(0, update)

    def start_training(self):
        if self.original is None:
            messagebox.showinfo(
                "Select Image", "Please choose an image first."
            )
            return

        if self.training:
            return

        self.training = True
        self.train_button.config(state="disabled")
        self.progress["value"] = 0

        threading.Thread(
            target=self.run_training, daemon=True
        ).start()

    def run_training(self):
        try:
            coords = self.coords
            colors = self.colors
            target = colors.numpy()

            # Baseline model
            self.set_status("Training Baseline MLP...")
            start = time.time()

            baseline_model = BaselineMLP()

            baseline_prediction = train_model(
                baseline_model, coords, colors, STEPS,
                self.set_status, "Baseline MLP"
            )

            btime = time.time() - start
            bmse, bpsnr = calculate_metrics(
                baseline_prediction, target
            )

            # Fourier model
            self.set_status("Training Fourier MLP...")
            start = time.time()

            fourier_model = FourierMLP()

            fourier_prediction = train_model(
                fourier_model, coords, colors, STEPS,
                self.set_status, "Fourier MLP"
            )

            ftime = time.time() - start
            fmse, fpsnr = calculate_metrics(
                fourier_prediction, target
            )

            # Metrics before correction
            before_prediction = fourier_prediction.copy()
            before_mse = fmse
            before_psnr = fpsnr

            # Adaptive self-correction
            self.set_status("Running adaptive self-correction...")
            start = time.time()

            corrected_prediction = self_correct_model(
                fourier_model, coords, colors, self.set_status
            )

            correction_time = time.time() - start

            after_mse, after_psnr = calculate_metrics(
                corrected_prediction, target
            )

            # Generate error heatmap
            target_img = target.reshape(SIZE, SIZE, 3)
            corrected_img = corrected_prediction.reshape(SIZE, SIZE, 3)

            errors = np.sqrt(
                np.mean((target_img - corrected_img) ** 2, axis=2)
            )

            errors /= max(float(errors.max()), 1e-8)

            heat = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            heat[:, :, 0] = (errors * 255).astype(np.uint8)
            heat[:, :, 1] = ((1 - errors) * 70).astype(np.uint8)

            heat_image = Image.fromarray(heat, "RGB")

            results = (
                prediction_to_image(baseline_prediction),
                prediction_to_image(before_prediction),
                prediction_to_image(corrected_prediction),
                heat_image,
                bmse, bpsnr, btime,
                fmse, fpsnr, ftime,
                before_mse, before_psnr,
                after_mse, after_psnr, correction_time
            )

            self.root.after(
                0, lambda result=results: self.show_results(*result)
            )

        except Exception as error:
            self.root.after(
                0,
                lambda msg=str(error): messagebox.showerror(
                    "Training Error", msg
                )
            )
            self.set_status("Training failed. Check the error message.")

        finally:
            self.root.after(0, self.finish_training)

    def show_results(
        self,
        baseline_img, fourier_img, corrected_img, heat_img,
        bmse, bpsnr, btime,
        fmse, fpsnr, ftime,
        before_mse, before_psnr,
        after_mse, after_psnr, correction_time
    ):
        self.display_image(
            self.baseline_view, baseline_img, "baseline"
        )
        self.display_image(
            self.fourier_view, fourier_img, "fourier"
        )
        self.display_image(
            self.heatmap_view, heat_img, "heatmap"
        )
        self.display_image(
            self.before_view, fourier_img, "before"
        )
        self.display_image(
            self.after_view, corrected_img, "after"
        )

        self.metrics.set(
            "BASELINE MLP\n"
            f"MSE: {bmse:.6f} | PSNR: {bpsnr:.2f} dB | "
            f"Time: {btime:.2f} sec\n\n"
            "FOURIER MLP\n"
            f"MSE: {fmse:.6f} | PSNR: {fpsnr:.2f} dB | "
            f"Time: {ftime:.2f} sec"
        )

        change = (
            (before_mse - after_mse) / max(before_mse, 1e-12)
        ) * 100

        if change > 0:
            outcome = f"MSE reduced by {change:.2f}%"
        elif change < 0:
            outcome = f"MSE increased by {abs(change):.2f}%"
        else:
            outcome = "MSE unchanged"

        self.correction_metrics.set(
            "BEFORE ADAPTIVE CORRECTION\n"
            f"MSE: {before_mse:.6f} | "
            f"PSNR: {before_psnr:.2f} dB\n\n"
            "AFTER ADAPTIVE CORRECTION\n"
            f"MSE: {after_mse:.6f} | "
            f"PSNR: {after_psnr:.2f} dB\n"
            f"{outcome} | Time: {correction_time:.2f} sec"
        )

        try:
            csv_path, _ = save_experiment_result(
                os.path.basename(self.image_path),
                bmse, bpsnr, btime,
                fmse, fpsnr, ftime,
                before_mse, before_psnr,
                after_mse, after_psnr, correction_time
            )

            self.status_text.set(
                f"Completed | {outcome} | Results saved to "
                f"{os.path.basename(csv_path)}"
            )

        except Exception as error:
            messagebox.showwarning(
                "CSV Save Error",
                f"Training finished, but CSV saving failed:\n{error}"
            )

        self.progress["value"] = 100

        # IMPORTANT: Show all metrics in a popup.
        messagebox.showinfo(
            "Neural Painter - Final Results",
            f"BASELINE MLP\n"
            f"MSE: {bmse:.6f}\n"
            f"PSNR: {bpsnr:.2f} dB\n"
            f"Training time: {btime:.2f} seconds\n\n"
            f"FOURIER MLP\n"
            f"MSE: {fmse:.6f}\n"
            f"PSNR: {fpsnr:.2f} dB\n"
            f"Training time: {ftime:.2f} seconds\n\n"
            f"BEFORE ADAPTIVE CORRECTION\n"
            f"MSE: {before_mse:.6f}\n"
            f"PSNR: {before_psnr:.2f} dB\n\n"
            f"AFTER ADAPTIVE CORRECTION\n"
            f"MSE: {after_mse:.6f}\n"
            f"PSNR: {after_psnr:.2f} dB\n\n"
            f"{outcome}\n"
            f"Correction time: {correction_time:.2f} seconds"
        )

    def finish_training(self):
        self.training = False
        self.train_button.config(state="normal")


# =============== RUN APPLICATION ===============
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralPainterApp(root)
    root.mainloop()