# Deployment Guide: ET-NeuralCast

This project is configured for **Static Site Deployment** on Netlify. 
The complex AI/ML processing happens on your local machine (using your GPU), and the resulting static visualization files (HTML, JSON, PNG) are uploaded to GitHub, which Netlify then serves to the world.

## 1. Initial Setup (One-Time)

### A. GitHub
Ensure your project is pushed to a GitHub repository:
```bash
git add .
git commit -m "Initial commit for deployment"
git push origin main
```

### B. Netlify
1.  Log in to [Netlify](https://app.netlify.com).
2.  Click **"Add new site"** -> **"Import from an existing project"**.
3.  Select **GitHub** and authorize it.
4.  Choose your repository (`S2S-Forecast-Project`).
5.  **Build Settings:**
    *   **Base directory**: Leave empty (root).
    *   **Build command**: Leave empty.
    *   **Publish directory**: Leave empty (serves root `index.html`).
6.  Click **Deploy Site**.

---

## 2. Routine Update Workflow

Whenever you generate new forecasts or re-train the model, follow these steps to update the live website.

### Step 1: Update Forecasts & Metrics
Run the local pipeline to generate new maps and plots:

```bash
# 1. Generate Forecasts (Maps & JSONs)
python src/inference/predict.py

# 2. Update Logging Charts (White Theme)
python src/evaluation/plot_mlflow.py

# 3. (Optional) Run Evaluation Metrics if you have new observed data
python src/evaluation/evaluate.py
```

### Step 2: Verify Locally
Open `index.html` in your browser to verify that the new maps and charts look correct.

### Step 3: Deploy to Live Site
Simply commit and push the updated `outputs/` folder. Netlify detects the push and updates the site within seconds.

```bash
git add outputs/
git commit -m "Update forecasts for [Date]"
git push
```

## 3. Configuration Details

*   **`netlify.toml`**: Located in the project root, this file configures Netlify to serve the root directory and allow Cross-Origin Resource Sharing (CORS) for the JSON data files.
*   **`.gitignore`**: Blocks heavy `*.nc` (NetCDF) data files but **allows** light `*.json`, `*.html`, and `*.png` files in `outputs/` to be synced.

## 4. Troubleshooting

*   **HTML file not updating?**
    *   Make sure you ran `git add outputs/` before verifying.
*   **Validation Plot Missing?**
    *   Ensure you ran `python src/evaluation/plot_mlflow.py` which handles the encoding fixes for the validation chart.
