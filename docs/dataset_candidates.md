# Candidate Datasets for S2S Forecasting in Ethiopia
**Domain**: Ethiopia (3°N - 15°N, 33°E - 48°E)

For S2S forecasting (2 weeks to 2 months ahead), you cannot rely solely on local data. You must capture global teleconnections (like ENSO and IOD) that "drive" the weather in East Africa.

## 1. Target Data (Ground Truth for $Y$)
These are the datasets you will try to predict. For Ethiopia, you need high-quality, long-term records.

| Variable | Dataset | Resolution | Why it's best for Ethiopia |
| :--- | :--- | :--- | :--- |
| **Precipitation** | **CHIRPS v2.0** | 0.05° (~5km) | **The Standard for East Africa.** Blends satellite IR with station data. highly accurate for complex topography like the Ethiopian Highlands. |
| **Precipitation** | **TAMSAT** | 0.0375° | Evaluation alternative. Very good for convective rain in Africa, useful to cross-validate against CHIRPS. |
| **Temperature** | **ERA5-Land** | 0.1° (~9km) | High-resolution reanalysis. Better than coarse global models for capturing temperature gradients in the highlands. |

## 2. Input Data (Predictors for $X$)
S2S skill comes from the ocean and large-scale atmospheric waves. You generally need **Global** or **Large Regional** coverage for these, not just the Ethiopia domain.

### A. Atmospheric Dynamics (ERA5 Reanalysis)
*Source: ECMWF (via `cdsapi`)*
*   **Geopotential Height (Z500)**: Tracks high/low pressure systems and Rossby waves.
*   **Zonal/Meridional Wind (U850, V850, U200)**: 850hPa for moisture transport (Somali Jet), 200hPa for the Tropical Easterly Jet (TEJ).
*   **Specific Humidity (Q)**: Moisture availability in the atmosphere.
*   **Velocity Potential (VP200)**: Proxy for the Madden-Julian Oscillation (MJO), a key S2S driver.

### B. Ocean Drivers (SST)
*Source: ERA5 or OISST v2*
*   **Sea Surface Temperature (SST)**: You specifically need to feed the model SSTs from:
    1.  **Indian Ocean**: The **Indian Ocean Dipole (IOD)** is the primary driver of Oct-Dec (short rains) in Ethiopia.
    2.  **Pacific Ocean**: **ENSO (El Niño/La Niña)** strongly affects the Jun-Sep (Kiremt) season.

### C. Land Surface
*   **Soil Moisture**: From ERA5-Land. Provides "memory" to the system (wet soil leads to more evaporation -> more rain).
*   **Vegetation Indices (NDVI)**: Can help in seasonal transition periods.

## 3. Recommended Data Engineering Strategy

### Domain Selection for Inputs
Do not crop inputs to Ethiopia only!
*   **Precipitation/Temp (Targets)**: Crop to **[3-15°N, 33-48°E]**.
*   **ERA5 Inputs (Predictors)**: Use a much larger domain, e.g., **[40°S - 40°N, 0°E - 100°E]**.
    *   This covers the Indian Ocean (for moisture source) and the Congo Basin (convection can propagate East).
    *   Alternatively, use global coarse-resolution (e.g., 2.5°) inputs to capture teleconnections efficiently.

### Temporal Resolution
*   **Raw Data**: Daily (or 6-hourly if storage permits).
*   **Model Input**: Aggregated to **Weekly** or **Bi-weekly** means. S2S models struggle with daily noise; predicting "Week 3 Average Rainfall" is much more successful.
