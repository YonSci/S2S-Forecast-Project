# ET-NeuralCast: Advanced S2S Forecasting for Ethiopia

[![Dashboard](https://img.shields.io/badge/Live-Dashboard-blue?style=for-the-badge&logo=github)](https://YonSci.github.io/S2S-Forecast-Project/)
[![Frequency](https://img.shields.io/badge/Update-Daily%20(00:00%20UTC)-brightgreen?style=for-the-badge&logo=github-actions)](.github/workflows/forecast.yml)
[![Architecture](https://img.shields.io/badge/Model-Hybrid%20U--Net%20GAN-orange?style=for-the-badge)](src/models/unet.py)

**ET-NeuralCast** is an enterprise-grade Sub-seasonal to Seasonal (S2S) forecasting framework. By fusing global atmospheric planetary drivers with high-resolution regional precipitation history, the system provides high-fidelity rainfall intelligence across Ethiopia with 0.05° spatial precision.

---

## 🎯 Objectives
*   **Bridge the Gap**: Transition coarse global climate data (1.0°) into localized regional reality (0.05°).
*   **Predict Extremes**: Use anomaly-based learning to identify deviations from the norm, critical for flood and drought early warning.
*   **Automate Insight**: Move scientific models out of notebooks and into an autonomous daily production pipeline.

---

## 📊 1. Data Pipeline

### a) Preprocessing & Normalization
The system ingests **ERA5** reanalysis and **CHIRPS** precipitation datasets. The preprocessing logic (`src/data/preprocessor.py`) performs:
*   **Spatial Focus**: Clipping to Ethiopia's bounding box.
*   **Temporal Aggregation**: Computing weekly means to align with S2S timescales.
*   **Lagged Predictors**: Creating temporal lags to capture atmospheric memory.
*   **Anomaly Extraction**: Subtracting weekly climatology (learned exclusively from the training set) to isolate significant atmospheric shifts.
*   **Standardization**: Applying zero-mean/unit-variance normalization using training-only statistics (`src/data/normalization.py`).

### b) Dataloader Implementation
The core is a custom `S2SDataset` (`src/data/dataloader.py`) which:
*   Loads tabular and gridded data from NetCDF.
*   Applies normalization on-the-fly.
*   Supports specific year selection and configurable forecast lead times.
*   Maps samples to the typical PyTorch `[Batch, Channel, H, W]` format.

### c) Data Format
*   **Input Tensors**: `[batch, time, variable, lat, lon]` (e.g., past 4 weeks, 5 variables, 48x60 grid).
*   **Target Tensors**: `[batch, target_window, 1, lat, lon]`.

---

## 🤖 2. Model Architecture
The core of ET-NeuralCast is a **Spatially-Aware Generative Adversarial Network (GAN)**:

### a) U-Net Generator (`src/models/unet.py`)
*   **Encoder**: Stacks `Conv2d` layers, downsampling while increasing channels (64→128→256→512), utilizing BatchNorm and LeakyReLU.
*   **Bottleneck**: Compresses features into a rich latent representation.
*   **Decoder**: Uses `TransposedConv` for upsampling with **Skip Connections** to corresponding encoder layers, preserving spatial resolution.
*   **Output**: Single-channel grid with `Tanh` activation to produce normalized precipitation anomalies.

### b) Discriminator (PatchGAN)
*   Receives both the input drivers and the prediction (or observation) map.
*   Outputs a grid of scores (e.g., 16x16) to evaluate local realism.
*   Encourages the generator to produce high-frequency details and physically plausible rainfall textures.

---

## 📉 3. Training Loop & Loss

### a) Two-Phase Training Strategy
1.  **Phase 1 (Warm Start)**: Training only the Generator using **L1 Loss** (Mean Absolute Error). this ensures stable learning of coarse rainfall patterns (`src/training/train_warmstart.py`).
2.  **Phase 2 (GAN Fine-tuning)**: Introducing the Discriminator. The Generator is trained to both minimize L1 error and "fool" the PatchGAN, resulting in sharper, more realistic outputs (`src/training/train_gan.py`).

### b) Loss Functions
*   **L1 Loss**: Primary loss to reduce absolute error and prevent outlier dominance.
*   **GAN Loss (MSE)**: Adversarial loss for PatchGAN style training.
*   **Spatial Weighting**: Capacity to emphasize specific regions of interest.

### c) Execution Strategy
*   **Chronological Splitting**: Ensuring no data leakage by following strict temporal order.
*   **Experiment Tracking**: Integration with MLflow for tracking parameters, metrics, and model checkpoints.

---

## 🚀 4. Usage & Operations

### Operational Pipeline
See `src/main_operational.py` for the end-to-end production workflow.
*   **Autonomous Operation**: Daily ingestion, preprocessing, inference, and dashboard deployment.
*   **Local Run**: `python src/main_operational.py`

### Testing & Training
*   **Data Prep**: `src/data/download_era5.py` & `download_chirps.py`
*   **Training**: Run the respective phase scripts in `src/training/`.

---

## 📂 Project Structure
```text
S2S-Forecast-Project/
├── .github/workflows/   <- Daily Automation (CI/CD)
├── checkpoints/         <- Saved Model & Normalizer
├── data/raw/            <- Ingested Atmospheric Data
├── outputs/             <- Current & Historical Forecasts
├── src/
│   ├── data/            <- Dataloaders & Normalization
│   ├── models/          <- U-Net GAN Architecture
│   ├── training/        <- Training Orchestration
│   ├── inference/       <- Prediction & Visualization
│   └── main_operational.py <- Production Wrapper
└── index.html           <- Dashboard UI (Light/Dark)
```

---

## ✉️ Contact
**Yonas Mersha**  
Hydro-Climate Modelling & ML/AI Expert  
International Livestock Research Institute (ILRI)

*   [📧 Email](mailto:yonas.mersha14@gmail.com)
*   [💼 LinkedIn](https://linkedin.com/in/yonas-mersha)
*   [💻 GitHub](https://github.com/YonSci)
*   [✍️ Medium](https://medium.com/@yonas.mersha14)
