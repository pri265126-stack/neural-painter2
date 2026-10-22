# Neural Painter 

### Image Reconstruction Using Neural Networks and Fourier Features

##  About the Project

Neural Painter is a machine learning project that explores how neural networks can reconstruct images from pixel coordinates.

The model takes the `(x, y)` coordinates of a pixel as input and predicts its RGB color values. By learning from image pixels during training, the network gradually reconstructs the original image.

The project compares a standard Multi-Layer Perceptron (MLP) with a Fourier-feature-based MLP to study how different architectures affect image reconstruction quality.

##  Objectives

- Reconstruct images using coordinate-based neural networks.
- Compare a baseline MLP with a Fourier-feature-based MLP.
- Experiment with different pixel sampling strategies.
- Evaluate reconstruction quality using Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).
- Analyze how model architecture and sampling strategies affect reconstruction performance.

##  How It Works

1. **Input:** Pixel coordinates `(x, y)` from an image.
2. **Model:** A neural network learns the relationship between pixel coordinates and RGB colors.
3. **Training:** The model predicts pixel colors and updates its parameters to reduce reconstruction error.
4. **Reconstruction:** The trained model predicts colors for image coordinates to generate a reconstructed image.
5. **Evaluation:** The reconstructed image is compared with the original image using MSE and PSNR.

##  Models Compared

### 1. Baseline MLP
A standard neural network that learns directly from pixel coordinates.

### 2. Fourier-Feature MLP
A neural network that uses Fourier feature encoding to represent coordinates before predicting RGB values. This helps investigate whether richer coordinate representations improve image reconstruction.

##  Experiments

The project explores:

- Baseline versus Fourier-feature reconstruction.
- Uniform, error-guided, and hybrid pixel sampling strategies.
- Different sampling ratios.
- Reconstruction across multiple test images.
- Repeated runs to examine performance consistency.

##  Evaluation Metrics

- **Mean Squared Error (MSE):** Measures the average squared difference between original and reconstructed pixel values. Lower is better.
- **Peak Signal-to-Noise Ratio (PSNR):** Measures reconstruction quality on a logarithmic scale. Higher is generally better when the same image and pixel-value range are used.

##  Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)

## Project Structure

```text
neural-painter2/
├── neural_painter.py
├── visual_reconstruction_comparison.py
├── consolidated_report.py
├── final_report.py
├── fourier_experiment.py
├── fourier_final_validation.py
├── fourier_multi_image_experiment.py
├── fourier_ratio_experiment.py
├── fourier_sampling_experiment.py
├── sampling_ratio_experiment.py
├── test_images/
├── experiment_outputs/
└── README.md
```

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/pri265126-stack/neural-painter2.git
cd neural-painter2
```

### 2. Install dependencies

```bash
pip install torch numpy pandas matplotlib pillow
```

### 3. Run the main experiment

```bash
python neural_painter.py
```

To run the visual reconstruction comparison:

```bash
python visual_reconstruction_comparison.py
```

##  Future Improvements

- Build an interactive interface for uploading custom images.
- Visualize training progress and reconstruction quality.
- Extend experiments to more images and compare model performance under different settings.

##  Project Status

This project is an experimental exploration of coordinate-based neural networks for image reconstruction, with a focus on model comparison, sampling strategies, and quantitative evaluation.
