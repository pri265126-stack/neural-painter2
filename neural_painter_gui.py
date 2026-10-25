
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

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============== MODELS ===============
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
            [torch.sin(projected), torch.cos(projected)], dim=-1
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
            nn.Sigmoid()
        )

    def forward(self, coords):
        return self.network(self.encoding(coords))


# =============== IMAGE PROCESSING ===============
def prepare_image(path):
    original = Image.open(path).convert("RGB")
    small = original.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32) / 255.0

    axis = np.linspace(-1, 1, SIZE, dtype=np.float32)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    coords = np.stack([x, y], axis=-1).reshape(-1, 2)

    return (
        original,
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(pixels.reshape(-1, 3), dtype=torch.float32)
    )


def calculate_metrics(prediction, target):
    mse = float(np.mean((prediction - target) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    return mse, psnr


def prediction_to_image(prediction):
    array = np.clip(prediction.reshape(SIZE, SIZE, 3), 0, 1)
    return Image.fromarray((array * 255).astype(np.uint8), "RGB")


# =============== CSV RESULT SAVING ===============
def save_experiment_result(
    image_name,
    baseline_mse, baseline_psnr, baseline_time,
    fourier_mse, fourier_psnr, fourier_time,
    before_mse, before_psnr,
    after_mse, after_psnr, correction_time
):
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_results.csv"
    )

    columns = [
        "Date_Time",
        "Image_Name",
        "Baseline_MSE",
        "Baseline_PSNR_dB",
        "Baseline_Time_sec",
        "Fourier_MSE",
        "Fourier_PSNR_dB",
        "Fourier_Time_sec",
        "Before_Correction_MSE",
        "Before_Correction_PSNR_dB",
        "After_Correction_MSE",
        "After_Correction_PSNR_dB",
        "Correction_Time_sec",
        "MSE_Improvement_Percent"
    ]

    improvement = (
        (before_mse - after_mse) / max(before_mse, 1e-12)
    ) * 100

    row = {
        "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image_Name": image_name,
        "Baseline_MSE": round(baseline_mse, 6),
        "Baseline_PSNR_dB": round(baseline_psnr, 2),
        "Baseline_Time_sec": round(baseline_time, 2),
        "Fourier_MSE": round(fourier_mse, 6),
        "Fourier_PSNR_dB": round(fourier_psnr, 2),
        "Fourier_Time_sec": round(fourier_time, 2),
        "Before_Correction_MSE": round(before_mse, 6),
        "Before_Correction_PSNR_dB": round(before_psnr, 2),
        "After_Correction_MSE": round(after_mse, 6),
        "After_Correction_PSNR_dB": round(after_psnr, 2),
        "Correction_Time_sec": round(correction_time, 2),
        "MSE_Improvement_Percent": round(improvement, 2)
    }

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()

        writer.writerow(row)

    return filename, improvement


# =============== TRAINING ===============
def train_model(model, coords, colors, steps, status, model_name):
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

        if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
            status(f"Training {model_name}: {step + 1}/{steps}")

    model.eval()
    with torch.no_grad():
        return model(coords).cpu().numpy()


def self_correct_model(model, coords, colors, status):
    # Give higher-error pixels more sampling probability.
    model.eval()

    with torch.no_grad():
        errors = torch.mean((model(coords) - colors) ** 2, dim=1)

    weights = errors.cpu().numpy() + 1e-8
    weights /= weights.sum()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE * 0.5
    )

    count = len(coords)
    model.train()

    for step in range(CORRECTION_STEPS):
        selected = np.random.choice(
            count,
            size=min(BATCH_SIZE, count),
            replace=True,
            p=weights
        )

        indices = torch.tensor(selected, dtype=torch.long)

        optimizer.zero_grad()
        output = model(coords[indices])
        loss = torch.mean((output - colors[indices]) ** 2)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0 or step == CORRECTION_STEPS - 1:
            status(
                f"Self-correction: {step + 1}/{CORRECTION_STEPS}"
            )

    model.eval()
    with torch.no_grad():
        return model(coords).cpu().numpy()


# =============== GUI ===============
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
            toolbar,
            text="Choose Image",
            command=self.choose_image
        ).pack(side="left", padx=5)

        self.file_label = ttk.Label(
            toolbar, text="No image selected"
        )
        self.file_label.pack(side="left", padx=10)

        self.train_button = ttk.Button(
            toolbar,
            text="Train & Reconstruct",
            command=self.start_training
        )
        self.train_button.pack(side="left", padx=5)

        self.status_text = tk.StringVar(
            value="Choose an image to begin."
        )
        ttk.Label(
            self.root,
            textvariable=self.status_text,
            anchor="center",
            padding=5
        ).pack(fill="x")

        self.progress = ttk.Progressbar(
            self.root, maximum=100, mode="determinate"
        )
        self.progress.pack(fill="x", padx=12, pady=5)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(
            fill="both", expand=True, padx=8, pady=8
        )

        self.original_tab = ttk.Frame(self.tabs)
        self.comparison_tab = ttk.Frame(self.tabs)
        self.heatmap_tab = ttk.Frame(self.tabs)
        self.correction_tab = ttk.Frame(self.tabs)

        self.tabs.add(self.original_tab, text="Original")
        self.tabs.add(self.comparison_tab, text="Model Comparison")
        self.tabs.add(self.heatmap_tab, text="Error Heatmap")
        self.tabs.add(self.correction_tab, text="Self-Correction")

        self.original_view = self.make_panel(
            self.original_tab, "Original Image", "top"
        )

        comparison_images = ttk.Frame(self.comparison_tab)
        comparison_images.pack(fill="both", expand=True)

        self.baseline_view = self.make_panel(
            comparison_images, "Baseline MLP", "left"
        )
        self.fourier_view = self.make_panel(
            comparison_images, "Fourier MLP", "left"
        )

        self.metrics = tk.StringVar(
            value="Model metrics will appear here after training."
        )
        ttk.Label(
            self.comparison_tab,
            textvariable=self.metrics,
            font=("Consolas", 11),
            justify="center",
            padding=8
        ).pack(side="bottom", fill="x")

        self.heatmap_view = self.make_panel(
            self.heatmap_tab, "Error Heatmap", "top"
        )

        correction_images = ttk.Frame(self.correction_tab)
        correction_images.pack(fill="both", expand=True)

        self.before_view = self.make_panel(
            correction_images,
            "Fourier Before Correction",
            "left"
        )
        self.after_view = self.make_panel(
            correction_images,
            "Fourier After Correction",
            "left"
        )

        self.correction_metrics = tk.StringVar(value="")
        ttk.Label(
            self.correction_tab,
            textvariable=self.correction_metrics,
            font=("Consolas", 11),
            justify="center",
            padding=8
        ).pack(side="bottom", fill="x")

    def make_panel(self, parent, title, side):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(
            side=side, fill="both", expand=True, padx=6, pady=6
        )

        label = ttk.Label(
            frame, text="Image will appear here", anchor="center"
        )
        label.pack(fill="both", expand=True)
        label.image = None
        return label

    def display_image(
        self, label, image, key, max_width=650, max_height=500
    ):
        image = image.copy().convert("RGB")
        width, height = image.size

        scale = min(
            max_width / width,
            max_height / height
        )

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
                text=f"{os.path.basename(path)} | Original size: "
                     f"{self.original.width} x {self.original.height}"
            )

            self.display_image(
                self.original_view,
                self.original,
                "original",
                1000,
                650
            )

            self.status_text.set(
                "Image loaded. Click Train & Reconstruct."
            )
            self.progress["value"] = 0

        except Exception as error:
            messagebox.showerror("Image Error", str(error))

    def set_status(self, text):
        self.root.after(
            0, lambda msg=text: self.status_text.set(msg)
        )

    def start_training(self):
        if self.original is None:
            messagebox.showinfo(
                "Select Image",
                "Please choose an image first."
            )
            return

        if self.training:
            return

        self.training = True
        self.train_button.config(state="disabled")
        self.progress["value"] = 0

        threading.Thread(
            target=self.run_training,
            daemon=True
        ).start()

    def run_training(self):
        try:
            coords = self.coords
            colors = self.colors
            target = colors.numpy()

            # Baseline training
            self.set_status("Training Baseline MLP...")
            start = time.time()

            baseline_model = BaselineMLP()
            baseline_prediction = train_model(
                baseline_model,
                coords,
                colors,
                STEPS,
                self.set_status,
                "Baseline MLP"
            )

            baseline_time = time.time() - start
            baseline_mse, baseline_psnr = calculate_metrics(
                baseline_prediction, target
            )

            # Fourier training
            self.set_status("Training Fourier MLP...")
            start = time.time()

            fourier_model = FourierMLP()
            fourier_prediction = train_model(
                fourier_model,
                coords,
                colors,
                STEPS,
                self.set_status,
                "Fourier MLP"
            )

            fourier_time = time.time() - start
            fourier_mse, fourier_psnr = calculate_metrics(
                fourier_prediction, target
            )

            before_prediction = fourier_prediction.copy()
            before_mse = fourier_mse
            before_psnr = fourier_psnr

            # Self-correction
            self.set_status("Running self-correction...")
            start = time.time()

            corrected_prediction = self_correct_model(
                fourier_model,
                coords,
                colors,
                self.set_status
            )

            correction_time = time.time() - start
            after_mse, after_psnr = calculate_metrics(
                corrected_prediction, target
            )

            # Error heatmap
            target_img = target.reshape(SIZE, SIZE, 3)
            corrected_img = corrected_prediction.reshape(
                SIZE, SIZE, 3
            )

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
                baseline_mse, baseline_psnr, baseline_time,
                fourier_mse, fourier_psnr, fourier_time,
                before_mse, before_psnr,
                after_mse, after_psnr, correction_time
            )

            self.root.after(
                0,
                lambda result=results: self.show_results(*result)
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
            self.baseline_view, baseline_img, "baseline", 650, 500
        )
        self.display_image(
            self.fourier_view, fourier_img, "fourier", 650, 500
        )
        self.display_image(
            self.heatmap_view, heat_img, "heatmap", 800, 600
        )
        self.display_image(
            self.before_view, fourier_img, "before", 650, 500
        )
        self.display_image(
            self.after_view, corrected_img, "after", 650, 500
        )

        self.metrics.set(
            "BASELINE MLP\n"
            f"MSE: {bmse:.6f} | PSNR: {bpsnr:.2f} dB"
            f" | Time: {btime:.2f} sec\n\n"
            "FOURIER MLP\n"
            f"MSE: {fmse:.6f} | PSNR: {fpsnr:.2f} dB"
            f" | Time: {ftime:.2f} sec"
        )

        change = (
            (before_mse - after_mse) /
            max(before_mse, 1e-12)
        ) * 100

        if change > 0:
            outcome = f"MSE reduced by {change:.2f}%"
        elif change < 0:
            outcome = f"MSE increased by {abs(change):.2f}%"
        else:
            outcome = "MSE unchanged"

        self.correction_metrics.set(
            "BEFORE SELF-CORRECTION\n"
            f"MSE: {before_mse:.6f} | PSNR: {before_psnr:.2f} dB\n\n"
            "AFTER SELF-CORRECTION\n"
            f"MSE: {after_mse:.6f} | PSNR: {after_psnr:.2f} dB\n"
            f"{outcome} | Time: {correction_time:.2f} sec"
        )

        # Save experiment automatically as a new CSV row.
        try:
            csv_path, _ = save_experiment_result(
                os.path.basename(self.image_path),
                bmse, bpsnr, btime,
                fmse, fpsnr, ftime,
                before_mse, before_psnr,
                after_mse, after_psnr,
                correction_time
            )

            self.status_text.set(
                f"Completed | {outcome} | Results saved to "
                f"{os.path.basename(csv_path)}"
            )

        except Exception as error:
            self.status_text.set(
                "Training completed, but CSV saving failed."
            )
            messagebox.showwarning(
                "CSV Save Error",
                f"Training finished, but results could not be saved:\n{error}"
            )

        self.progress["value"] = 100

    def finish_training(self):
        self.training = False
        self.train_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralPainterApp(root)
    root.mainloop()
