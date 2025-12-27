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

            # --- Plot C: Validation Performance ---
            try:
                h_rmse = client.get_metric_history(run_id, "val_rmse")
                h_acc = client.get_metric_history(run_id, "val_acc")
                
                if h_rmse and h_acc:
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # RMSE (Left Axis)
                    steps_rmse = [m.step for m in h_rmse]
                    vals_rmse = [m.value for m in h_rmse]
                    l1 = ax1.plot(steps_rmse, vals_rmse, 'b-o', label='Val RMSE', linewidth=2)
                    ax1.set_xlabel("Epochs", fontsize=12)
                    ax1.set_ylabel("RMSE (mm/day)", color='blue', fontsize=12)
                    ax1.tick_params(axis='y', labelcolor='blue')
                    ax1.grid(True, alpha=0.3)
                    
                    # ACC (Right Axis)
                    ax2 = ax1.twinx()
                    steps_acc = [m.step for m in h_acc]
                    vals_acc = [m.value for m in h_acc]
                    l2 = ax2.plot(steps_acc, vals_acc, 'orange', marker='s', linestyle='-', label='Val ACC', linewidth=2)
                    ax2.set_ylabel("Anomaly Correlation (ACC)", color='orange', fontsize=12)
                    ax2.tick_params(axis='y', labelcolor='orange')
                    
                    # Combined Title & Legend
                    lines = l1 + l2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc='center right')
                    
                    plt.title("GAN Validation Performance (Unseen Years)", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, "logs_gan_validation.png"), dpi=300)
                    plt.close()
            except Exception as e:
                print(f"Skipping validation plot: {e}")

if __name__ == "__main__":
    plot_experiment_logs()
