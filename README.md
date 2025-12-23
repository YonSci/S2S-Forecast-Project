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

## 🤖 Model Architecture
The core of ET-NeuralCast is a **Spatially-Aware Generative Adversarial Network (GAN)**:

*   **Generator (U-Net)**: A deep encoder-decoder architecture with skip-connections. It learns to reconstruct high-resolution precipitation structures by analyzing global moisture transport and pressure surfaces.
*   **Discriminator (PatchGAN)**: A convolutional critic that evaluates the "physical plausibility" of the generated maps against historical CHIRPS topography, forcing the generator to produce sharp, realistic rainfall patterns.
*   **Anomaly-Centric Learning**: The model targets **Precipitation Anomalies** (deviation from a 20-year mean) rather than raw values, reducing bias and focusing the neural network on significant atmospheric shifts.

---

## ⚙️ Operational Pipeline
The system operates on a fully autonomous **Daily Flux Pipeline**:

1.  **Data Ingestion**: Automated triggers fetch real-time atmospheric drivers (ERA5/ERA5T) from the Copernicus CDS API.
2.  **Harmonization**: Global variables (Z500, Q700, Wind U/V, SST) are normalized using a production-synced scaling layer.
3.  **Inference Engine**: The U-Net GAN prepares four multi-perspective forecasts:
    *   **Anomaly**: Physical deviation from expectations.
    *   **Total Rain**: Absolute water volume reconstructed from regional baselines.
    *   **Percent of Normal**: Relative moisture impact.
    *   **Tercile Categories**: Categorical outlook (Below, Near, or Above Normal).
4.  **Edge Delivery**: Results are committed to the repository and deployed immediately to the [GitHub Pages Dashboard](https://YonSci.github.io/S2S-Forecast-Project/).

---

## 🛠️ Installation & Setup

### 1. Requirements
*   Python 3.10+
*   [Conda](https://docs.conda.io/en/latest/miniconda.html) or Micromamba (Recommended)
*   CDS API Key (For data acquisition)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/YonSci/S2S-Forecast-Project.git
cd S2S-Forecast-Project

# Create environment
conda create -n et_neuralcast python=3.10 -y
conda activate et_neuralcast

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install xarray netcdf4 cdsapi matplotlib cartopy shapely numpy
```

### 3. API Configuration
Create a `.cdsapirc` file in your home directory or set environment variables:
```bash
export CDSAPI_URL='https://cds.climate.copernicus.eu/api/v2'
export CDSAPI_KEY='YOUR_UID:YOUR_API_KEY'
```

---

## 🚀 Running the Workflow

### Local Operational Run
To run the same pipeline that powers the dashboard:
```bash
python src/main_operational.py
```
This script will:
1. Download recent ERA5 data.
2. Run GAN inference.
3. Generate `.png` maps and `.nc` data exports in the `outputs/` folder.

### Local Testing/Training
The project is modularized into dedicated phases:
*   **Data Prep**: `src/data/download_era5.py` & `download_chirps.py`
*   **Training**: `src/training/train_warmstart.py` (L1 Warmstart) & `train_gan.py` (GAN Refinement)

---

## 📊 Data Specifications
| Aspect | Input (Predictors) | Target (Predictand) |
| :--- | :--- | :--- |
| **Source** | ERA5 / ERA5T Reanalysis | CHIRPS v2.0 |
| **Grid** | 1.0° x 1.0° (Global) | 0.05° x 0.05° (Ethiopia) |
| **Variables** | Z, Q, U, V (200, 500, 850 hPa) + SST | Precipitation Anomaly |
| **Horizon** | S2S (Lead: 1-4 Weeks) | Static Climatology Baseline |

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
