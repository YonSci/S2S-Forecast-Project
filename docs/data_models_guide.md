# Data Formats and Deep Learning Architectures for S2S

## 1. The Recommended Data Format: NetCDF & Tensors
For all "Gridded" weather data, **output everything as NetCDF (.nc) files**.
*   **Why?**: It preserves coordinates (Lat/Lon/Time). If you save as CSV or NPY, you lose the spatial reference, which is a nightmare for plotting later.
*   **Tensor conversion**: During training, your `DataLoader` will convert these NetCDF files into 5D Tensors.

### The "Golden Shape": `(N, T, C, H, W)`
Deep Learning frameworks (PyTorch) expect data in specific dimensions.
*   **N (Batch Size)**: Number of samples per training step (e.g., 32).
*   **T (Time)**: Sequence length (e.g., past 4 weeks).
*   **C (Channels)**: The variable count (e.g., SST, Wind, Humidity = 3 channels).
*   **H (Height)**: Latitude dimension.
*   **W (Width)**: Longitude dimension.

---

## 2. Input ($X$) vs Target ($Y$) Formatting

### A. The Input ($X$)
*   **Format**: 3D or 4D Grid (Time + Lat + Lon).
*   **Resolution**: Can be coarser (e.g., 1.0° or 0.25°) to save memory.
*   **Variables (Channels)**: Stack multiple variables like layers in an image.
    *   *Channel 0*: Geopotential Height
    *   *Channel 1*: SST
    *   *Channel 2*: Soil Moisture
*   **Example Shape**: `[Batch, 4_weeks, 5_vars, 64_lat, 128_lon]`

### B. The Target ($Y$)
*   **Format**: 2D Grid (Lat + Lon) for a specific future window.
*   **Resolution**: High resolution (e.g., CHIRPS 0.05°) for Ethiopia.
*   **Example Shape**: `[Batch, 1_target_window, 1_var, 120_lat, 150_lon]`

---

## 3. How Models Use This Data

### 1. CNNs (Convolutional Neural Networks)
*   **Concept**: Treat the global weather map at time $t$ like a photograph.
*   **How it works**:
    *   The model slides filters over the global map to visually identify patterns like "High Pressure Ridge" or "El Niño Blob".
    *   **2D CNN**: Ignores time, just looks at a snapshot.
    *   **3D CNN**: Slides filters over Time, Lat, and Lon simultaneously. Captures motion (e.g., a storm moving East).

### 2. Encoder-Decoder (e.g., U-Net)
*   **Concept**: Image-to-Image translation.
*   **Input**: Global Coarse Atmosphere.
*   **Output**: Local High-Res Rainfall over Ethiopia.
*   **Mechanism**:
    *   **Encoder**: Compresses the big global picture into a small "feature vector" (understanding the context).
    *   **Decoder**: Upscales that vector back into a detailed map of Ethiopia rainfall.
*   **Best for**: Downscaling and capturing local topography effects.

### 3. ConvLSTM / ConvGRU (Spatiotemporal)
*   **Concept**: A Recurrent Network (RNN) that has "eyes".
*   **How it works**:
    *   Standard LSTM takes a 1D vector (numbers) sequence.
    *   **ConvLSTM** takes a sequence of **Images**.
    *   Inside the LSTM cell, matrix multiplication is replaced by **Convolutions**.
    *   It remembers the "state" of the weather (e.g., "Soil is wet from yesterday", "Jet stream is blocking").
*   **Best for**: Video prediction tasks where evolution over time is critical.

### 4. GANs (Generative Adversarial Networks)
*   **Concept**: Solving the "Blurry Forecast" problem.
*   **Problem**: MSE/RMSE loss encourages models to predict the *average* (blurry drizzle everywhere). Real rain is sharp and patchy.
*   **Mechanism**:
    *   **Generator**: Tries to predict the rain map.
    *   **Discriminator**: A critic that looks at the prediction and says "Fake! This looks too smooth to be real CHIRPS data."
*   **Result**: The Generator is forced to create sharp, realistic storms to fool the Discriminator.
*   **Best for**: Extreme event prediction and probabilistic forecasting.

## 4. Summary Table

| Model Type   | Input Shape       | Good For...                  | Weakness               |
| :----------- | :---------------- | :--------------------------- | :--------------------- |
| **2D CNN**   | `(N, C, H, W)`    | Simple Pattern Recognition   | Ignores time history   |
| **3D CNN**   | `(N, C, T, H, W)` | Short-term motion            | Computationally heavy  |
| **ConvLSTM** | `(N, T, C, H, W)` | Evolution / Memory           | Hard to train, slow    |
| **U-Net**    | `(N, C, H, W)`    | Downscaling / Spatial Detail | Fixed window size      |
| **GAN**      | `(...)`           | Sharpness / Extremes         | Hard/Unstable training |
