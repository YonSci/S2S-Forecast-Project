# Operational S2S Strategy: From Training to Live Forecasting

## The Challenge
You plan to train your model using **ERA5 (Reanalysis)** as inputs ($X$) and **CHIRPS** as targets ($Y$).
*   **ERA5** is perfect for training because it is consistent and historical (1980-Present).
*   **BUT**: ERA5 is **NOT available in real-time**. It has a latency of ~5 days for the preliminary version (ERA5T) and months for the quality-controlled version.

To make this operational, you need **Real-Time Inputs** that "look like" ERA5 but are available *today*.

---

## Option 1: The "Direct S2S" Approach (Pure AI Forecast)
*Your model acts as the simulator. It takes Today's State $\to$ Predicts Week 3-4.*

### 1. Atmospheric Inputs (Wind, Geopotential, Humidity)
Since ERA5 is produced by ECMWF, the best operational proxy is the **ECMWF Operational Analysis (IFS)**.
*   **Source**: ECMWF Open Data or commercial license.
*   **Alternative (Free)**: **GFS Analysis (NCEP)**.
    *   *Risk*: GFS data looks slightly different from ERA5. This "Domain Shift" can confuse a Neural Network.
    *   *Solution*: If you use GFS for operations, you should ideally **train on GFS Reforecasts** or fine-tune your ERA5-trained model on GFS data.

### 2. Ocean Inputs (SST)
The ocean changes slowly, so having the absolute latest hour is less critical, but you still need daily updates.
*   **Operational Source**: **NOAA OISST v2 (Optimum Interpolation SST)**.
    *   **Availability**: 1-day delay (Near Real-Time).
    *   **Consistency**: OISST is often used to *drive* reanalyses, so the consistency with ERA5 SST is generally high.

---

## Option 2: The "Hybrid" Approach (Model Output Statistics / Downscaling)
*Your model corrects a Physical Model. Input is a coarse Physical Forecast $\to$ Predicts High-Res Reality.*

In this scenario, you do **not** train on ERA5. Instead, you train on **Reforecasts (Hindcasts)** from a dynamical model.
*   **Training Input ($X$)**: ECMWF S2S Reforecasts (past 20 years).
*   **Training Target ($Y$)**: CHIRPS.
*   **Operational Input**: Live ECMWF S2S Forecast.

**Verdict**: Since you specifically asked about **Training on ERA5**, you are likely following **Option 1 (Direct S2S)**.

---

## Recommended Operational Pipeline (for Option 1)

| Variable                          | Training Data (Historical) | Operational Data (Live)                     | Notes                                                                                           |
| :-------------------------------- | :------------------------- | :------------------------------------------ | :---------------------------------------------------------------------------------------------- |
| **Atmosphere** (Wind, Z500, etc.) | **ERA5 Reanalysis**        | **ERA5T** (Preliminary) or **GFS Analysis** | ERA5T has ~5 day lag. If you need *immediate* runs, use GFS Analysis but verify "Domain Shift". |
| **SST** (ENSO, IOD)               | **ERA5 SST**               | **NOAA OISST v2**                           | High consistency. Regrid OISST to match your ERA5 grid.                                         |
| **Precipitation** (Ground Truth)  | **CHIRPS v2.0**            | **CHIRPS-Prelim**                           | CHIRPS releases a preliminary version rapidly for monitoring.                                   |

### Handling the "Lag" (The 5-day Gap)
If you use **ERA5T** (5-day delay) for operations:
*   Your operational forecast for "Week 3" will actually be a forecast for "Week 3, starting 5 days ago".
*   This effectively means you are doing a **Lead 25-day forecast** to get a "Week 3" prediction relative to *today*.
*   **Recommendation**: For S2S (where lead times are long), a 5-day data lag is often acceptable. Using ERA5T is the safest bet for maintaining model accuracy because it comes from the exact same system as your training data.

### Summary Checklist for Operations
1.  **Download ERA5T**: Set up a cron job to fetch the latest preliminary ERA5 data (last month).
2.  **Download OISST**: Fetch the daily SST update.
3.  **Preprocessing**: Apply the **exact same** normalization (Mean/Std) you saved during training. *Do not re-calculate statistics on the live data.*
4.  **Inference**: Run the U-Net/cGAN.
