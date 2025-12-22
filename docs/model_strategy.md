# S2S Forecasting Model Training Strategy

## 1. Data Splitting Strategy
For Sub-seasonal to Seasonal (S2S) forecasting, strictly chronological splitting is critical to avoid data leakage (future information influencing past predictions) and to simulate real-world operational forecasting.

### Recommended Split (Example for a 20-year dataset)
* **Training Set (70%)**: First 14 years. Used to learn weights.
* **Validation Set (15%)**: Next 3 years. Used for:
    * Hyperparameter tuning (learning rate, layers, dropout).
    * Early stopping (to prevent overfitting).
    * Model selection.
* **Test Set (15%)**: Final 3 years. Used **only once** for final performance evaluation.

**Important**: Ensure a "purge" period between sets equal to your maximum forecast lead time (e.g., if predicting 4 weeks out, leave a 4-week gap) to ensure no target overlap.

## 2. Input & Target Design
* **Inputs ($X$)**:
    * Global/Regional Climate variables (SST, Geopotential Height, Soil Moisture, etc.) at time $t$.
    * Lagged variables (conditions at $t-1$ week, $t-2$ weeks).
    * Static features (Topography, Land/Sea mask).
    * Month/Day encoding (Cycle embeddings).
* **Targets ($Y$)**:
    * Aggregated Precipitation/Temperature at specific lead times (e.g., Week 3-4 average).
    * **Anomaly Prediction**: It is usually better to predict anomalies (deviations from the long-term mean) rather than raw values, as this removes the strong seasonal cycle which models can easily memorize but fail to outperform.

## 3. Training Workflow

### A. Preprocessing
1.  **Climatology Removal**: Calculate daily/weekly climatology from the *Training Set only*. Subtract this from Train, Val, and Test to create anomalies.
2.  **Normalization**: Standardize inputs (Zero Mean, Unit Variance) based on *Training Set statistics*.

### B. Loss Functions
Since S2S prediction is highly uncertain, consider probabilistic loss functions:
*   **MSE / RMSE**: Standard for deterministic regression (point forecasts).
*   **CRPS (Continuous Ranked Probability Score)**: If predicting a distribution (good for capturing uncertainty).
*   **Quantile Loss**: For predicting specific percentiles (e.g., 90th percentile for extreme events).

### C. Validation Strategy: Walk-Forward (Rolling Origin)
Instead of static k-fold CV, use **Walk-Forward Validation**:
1.  Train on Year 1-10 -> Validate on Year 11.
2.  Train on Year 1-11 -> Validate on Year 12.
3.  ...
This is computationally expensive but most robust for time-series. For deep learning, a single static chronological split is often accepted if the dataset is large enough.

## 4. Evaluation Metrics
Compare your DL model against standard baselines:
1.  **Climatology**: Predicting the historical average (Anomaly = 0).
2.  **Persistence**: Predicting that the next period will be the same as the current period.

Key Metrics:
*   **RMSE**: Root Mean Square Error (Magnitude of error).
*   **ACC (Anomaly Correlation Coefficient)**: Spatial correlation between predicted and observed anomalies. **This is the gold standard for S2S.**
    *   ACC = 1.0 (Perfect)
    *   ACC = 0.0 (No skill, same as climatology)
    *   ACC > 0.6 (Generally considered "useful" skill)

## 5. Experiment Tracking (MLflow)
*   Log parameters: `learning_rate`, `batch_size`, `dropout`, `model_architecture`.
*   Log metrics: `train_loss`, `val_loss`, `val_acc`, `val_rmse`.
*   Save artifacts: Model weights (`.pt`), plots of validation predictions vs. observations.
