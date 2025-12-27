import mlflow
import pandas as pd
import os

# Connect to the local DB
mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("Searching experiments...")
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"ID: {exp.experiment_id}, Name: {exp.name}")
    
    # Get runs
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if not runs.empty:
        print(f"Found {len(runs)} runs.")
        best_run = runs.sort_values("metrics.val_rmse_epoch", ascending=True).iloc[0] if "metrics.val_rmse_epoch" in runs.columns else runs.iloc[0]
        print(f"Best Run ID: {best_run.run_id}")
        
        # Try to fetch history for a metric
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(best_run.run_id, "train_loss_epoch")
        print(f"Train Loss History Steps: {len(history)}")
