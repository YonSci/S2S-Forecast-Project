# S2S Forecast Project: Full Training & Evaluation Lifecycle

This guide provides the step-by-step commands to retrain the model from scratch, generate forecasts, and deploy the workflow using **Windows Command Prompt**.

## The Strategy
1.  **Training Set (2000-2017)**: Used to teach the model patterns.
2.  **Validation Set (2018-2019)**: Used to tune the model and select the best checkpoint.
3.  **Test Set (2020-2024)**: "Unseen" data. Used ONLY for the final report.

---

## Step 1: Cleanup Old Outputs
Remove previous outputs to ensure a fresh start.

```cmd
del /Q outputs\*.json outputs\*.png outputs\*.html outputs\*.nc
```

## Step 2: Phase 1 Training (Warm-Start)
**Goal**: Teach the U-Net the "coarse" physics.
**Data**: Trains on **2000-2017**.

```cmd
python src/training/train_warmstart.py --years 2000-2017 --epochs 50 --lead 1
```

## Step 3: Phase 2 Training (GAN Fine-Tuning)
**Goal**: Add texture and realistic details.
**Data**: Train: **2000-2017**, Val: **2018-2019**.

```cmd
python src/training/train_gan.py --years 2000-2017 --val-years 2018-2019 --epochs 30 --lead 1
```

## Step 4: Final Evaluation (Metrics & Report)
**Goal**: Generate all statistical metrics, scatter plots, and confusion matrices for the **Evaluation Page** (`evaluation.html`).
**Data**: Test: **2020-2024**.

```cmd
python src/evaluation/evaluate.py --model checkpoints/G_best_acc_W1.pth --years 2020-2024 --lead 1
```

## Step 5: Generate Map Overlays (Required)
**Goal**: Create the transparent boundary layers (Country, Regions) for the interactive maps.
**Note**: This is CRITICAL because the cleanup step deleted these images.

```cmd
python src/visualization/create_overlays.py
```

## Step 6: Generate Forecast Maps (Dashboard)
**Goal**: Generate the visuals for the Home and Forecast pages.

```cmd
python src/inference/predict.py
```

## Step 7: Update Logs & Visuals
**Goal**: Regenerate the training log charts (White Theme) for `logs.html`.

```cmd
python src/evaluation/plot_mlflow.py
```

## Step 8: Deploy Changes
**Goal**: Push the new outputs to GitHub/Netlify.

```cmd
git add outputs/
git commit -m "Update model and forecasts"
git push
```
*(See `DEPLOY.md` for full deployment details)*
