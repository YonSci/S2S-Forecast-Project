import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT_DIR = "outputs"
DB_URI = "sqlite:///mlflow.db"

def smooth(scalars, weight=0.9):  # Weight between 0 and 1
    if not scalars: return []
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_experiment_logs():
    mlflow.set_tracking_uri(DB_URI)
    client = mlflow.tracking.MlflowClient()
    
    # ==========================================
    # 1. Warmstart Experiment
    # ==========================================
    warmstart_exp = mlflow.get_experiment_by_name("ET-NeuralCast-Warmstart")
    if warmstart_exp:
        runs = mlflow.search_runs(experiment_ids=[warmstart_exp.experiment_id], order_by=["attribute.start_time DESC"])
        if not runs.empty:
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            print(f"Plotting Warmstart logs for run {run_id}...")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Batch L1 Loss
            try:
                h_batch = client.get_metric_history(run_id, "batch_L1_loss")
                if h_batch:
                    steps = [m.step for m in h_batch]
                    vals = [m.value for m in h_batch]
                    smoothed = smooth(vals, 0.99)
                    ax.plot(steps, vals, alpha=0.15, color='lightblue', label='Batch L1 (Raw)')
                    ax.plot(steps, smoothed, color='blue', linewidth=1.5, label='Batch L1 (Smoothed)')
            except: pass

            # Epoch Avg Loss
            try:
                h_epoch = client.get_metric_history(run_id, "avg_L1_loss")
                if h_epoch:
                    steps = [m.step for m in h_epoch] # These steps might be epoch numbers or batch steps depending on logging
                    vals = [m.value for m in h_epoch]
                    # Plot epoch markers as red dots
                    ax.plot(steps, vals, 'ro-', linewidth=2, label='Epoch Avg L1')
            except: pass
            
            ax.set_title("Warmstart Training: L1 Reconstruction Loss", fontsize=16)
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("L1 Loss", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "logs_warmstart.png"), dpi=300)
            plt.close()

    # ==========================================
    # 2. GAN Experiment
    # ==========================================
    gan_exp = mlflow.get_experiment_by_name("ET-NeuralCast-GAN")
    if gan_exp:
        runs = mlflow.search_runs(experiment_ids=[gan_exp.experiment_id], order_by=["attribute.start_time DESC"])
        if not runs.empty:
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            print(f"Plotting GAN logs for run {run_id}...")
            
            # --- Plot A: Batch Level Dynamics ---
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # G_L1_loss
            try:
                h = client.get_metric_history(run_id, "G_L1_loss")
                if h:
                    steps = [m.step for m in h]
                    vals = [m.value for m in h]
                    smoothed = smooth(vals, 0.99)
                    ax.plot(steps, smoothed, color='blue', label='Gen L1 Loss')
            except: pass
            
            # G_GAN_loss
            try:
                h = client.get_metric_history(run_id, "G_GAN_loss")
                if h:
                    steps = [m.step for m in h]
                    vals = [m.value for m in h]
                    smoothed = smooth(vals, 0.99)
                    ax.plot(steps, smoothed, color='green', label='Gen Adversarial Loss')
            except: pass
            
            # D_loss
            try:
                h = client.get_metric_history(run_id, "D_loss")
                if h:
                    steps = [m.step for m in h]
                    vals = [m.value for m in h]
                    smoothed = smooth(vals, 0.99)
                    ax.plot(steps, smoothed, color='red', alpha=0.7, label='Discriminator Loss')
            except: pass
            
            ax.set_title("GAN Training Dynamics (Batch Level - Smoothed)", fontsize=16)
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "logs_gan_batch.png"), dpi=300)
            plt.close()

            # --- Plot B: Epoch Level Averages ---
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metrics = {
                "epoch_G_L1_avg": ("Gen L1 (Avg)", "blue", "o-"),
                "epoch_G_GAN_avg": ("Gen Adv (Avg)", "green", "s-"),
                "epoch_D_avg": ("Discrim (Avg)", "red", "^-")
            }
            
            for key, (label, color, marker) in metrics.items():
                try:
                    h = client.get_metric_history(run_id, key)
                    if h:
                        steps = [m.step for m in h]
                        vals = [m.value for m in h]
                        ax.plot(steps, vals, marker, color=color, label=label)
                except: pass

            ax.set_title("GAN Training Progress (Epoch Averages)", fontsize=16)
            ax.set_xlabel("Epochs", fontsize=12)
            ax.set_ylabel("Average Loss", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "logs_gan_epoch.png"), dpi=300)
            plt.close()

if __name__ == "__main__":
    plot_experiment_logs()
