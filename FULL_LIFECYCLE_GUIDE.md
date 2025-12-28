# S2S Forecst Project: Full Training & Evaluation Lifecycle

This guide provides the step-by-step commands to retrain the model from scratch using a scientifically rigorous **3-Way Data Split** to prevent data leakage.

## The Strategy
1.  **Training Set (2000-2017)**: Used to teach the model patterns.
2.  **Validation Set (2018-2019)**: Used to tune the model and select the best checkpoint (prevent overfitting).
3.  **Test Set (2020-2024)**: "Unseen" data. Used ONLY for the final report.

---

## Step 1: Cleanup Old Outputs
Remove previous evaluation reports, maps, and checkpoints to ensure a fresh start.

```powershell
# Remove old evaluation outputs
Remove-Item outputs\*.json -ErrorAction SilentlyContinue
Remove-Item outputs\*.png -ErrorAction SilentlyContinue

# (Optional) Remove old checkpoints for a clean start
# Remove-Item checkpoints\*.pth -ErrorAction SilentlyContinue
```

## Step 2: Phase 1 Training (Warm-Start)
**Goal**: Teach the U-Net the "coarse" physics using L1 Loss.
**Why?**: Starting a GAN from scratch is unstable. We first teach the Generator to make reasonable predictions (blurry but accurate average) before making it compete against a Discriminator.
*   **Data**: Trains on **2000-2017**.

```powershell
python src/training/train_warmstart.py --years 2000-2017 --epochs 50 --lead 1
```

## Step 3: Phase 2 Training (GAN Fine-Tuning)
**Goal**: Add texture and realistic details using Adversarial Loss.
**Logic**: We load the pre-trained U-Net from Phase 1 and continue training on the **same data**, but now the Discriminator forces it to generate sharper, more realistic maps.
*   **Train Data**: **2000-2017** (The model continues learning from this).
*   **Validation Data**: **2018-2019** (Used to pick the "Best Model" and stop training).

```powershell
python src/training/train_gan.py --years 2000-2017 --val-years 2018-2019 --epochs 30 --lead 1
```

## Step 4: Final Evaluation (The Test)
**Goal**: Assess performance on completely unseen years.
*   **Test Data**: **2020-2024** (The model has NEVER seen these years during Phase 1 or Phase 2).

```powershell
python src/evaluation/evaluate.py --model checkpoints/G_best_acc_W1.pth --years 2020-2024 --lead 1
```

## Step 5: Generate Overlays
Ensure the map overlays (Country, Regions) align perfectly with the new maps.

```powershell
python src/visualization/create_overlays.py
```

## Step 6: View Result
Open `evaluation.html` in your browser.
