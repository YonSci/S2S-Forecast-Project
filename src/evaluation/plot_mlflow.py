import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

            # --- Interactive Plotly Export ---
            try:
                fig_p = go.Figure()
                
                h_batch = client.get_metric_history(run_id, "batch_L1_loss")
                if h_batch:
                    steps = [m.step for m in h_batch]
                    vals = [m.value for m in h_batch]
                    smoothed = smooth(vals, 0.99)
                    
                    fig_p.add_trace(go.Scatter(x=steps, y=vals, mode='lines', name='Batch L1 (Raw)', 
                                             line=dict(color='lightblue', width=1), opacity=0.3))
                    fig_p.add_trace(go.Scatter(x=steps, y=smoothed, mode='lines', name='Batch L1 (Smoothed)',
                                             line=dict(color='deepskyblue', width=2)))

                h_epoch = client.get_metric_history(run_id, "avg_L1_loss")
                if h_epoch:
                    steps = [m.step for m in h_epoch]
                    vals = [m.value for m in h_epoch]
                    fig_p.add_trace(go.Scatter(x=steps, y=vals, mode='lines+markers', name='Epoch Avg L1',
                                             line=dict(color='red', width=3), marker=dict(size=8)))

                fig_p.update_layout(title="Warmstart Training: L1 Reconstruction Loss",
                                   xaxis_title="Steps", yaxis_title="L1 Loss",
                                   hovermode="x unified",
                                   font=dict(color="white"),
                                   paper_bgcolor='#1e293b',
                                   plot_bgcolor='#1e293b',
                                   xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                                   yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                fig_p.write_html(os.path.join(OUTPUT_DIR, "logs_warmstart.html"))
                print("Saved logs_warmstart.html")
            except Exception as e:
                print(f"Error creating Plotly warmstart: {e}")

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

            # --- Plotly Batch Dynamics ---
            try:
                fig_p = go.Figure()
                
                h_gl1 = client.get_metric_history(run_id, "G_L1_loss")
                if h_gl1:
                    steps = [m.step for m in h_gl1]
                    vals = [m.value for m in h_gl1]
                    smoothed = smooth(vals, 0.99)
                    fig_p.add_trace(go.Scatter(x=steps, y=smoothed, mode='lines', name='Gen L1 Loss', line=dict(color='blue')))
                    
                h_gan = client.get_metric_history(run_id, "G_GAN_loss")
                if h_gan:
                    steps = [m.step for m in h_gan]
                    vals = [m.value for m in h_gan]
                    smoothed = smooth(vals, 0.99)
                    fig_p.add_trace(go.Scatter(x=steps, y=smoothed, mode='lines', name='Gen Adv Loss', line=dict(color='green')))
                
                h_d = client.get_metric_history(run_id, "D_loss")
                if h_d:
                    steps = [m.step for m in h_d]
                    vals = [m.value for m in h_d]
                    smoothed = smooth(vals, 0.99)
                    fig_p.add_trace(go.Scatter(x=steps, y=smoothed, mode='lines', name='Discrim Loss', line=dict(color='red')))

                fig_p.update_layout(title="GAN Training Dynamics (Batch Level - Smoothed)",
                                   xaxis_title="Steps", yaxis_title="Loss",
                                   hovermode="x unified",
                                   font=dict(color="white"),
                                   paper_bgcolor='#1e293b',
                                   plot_bgcolor='#1e293b',
                                   xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                                   yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                fig_p.write_html(os.path.join(OUTPUT_DIR, "logs_gan_batch.html"))
                print("Saved logs_gan_batch.html")
            except Exception as e:
                print(f"Error creating Plotly GAN batch: {e}")

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

            # --- Plotly Epoch Averages ---
            try:
                fig_p = go.Figure()
                
                metrics = {
                    "epoch_G_L1_avg": ("Gen L1 (Avg)", "blue", "circle"),
                    "epoch_G_GAN_avg": ("Gen Adv (Avg)", "green", "square"),
                    "epoch_D_avg": ("Discrim (Avg)", "red", "triangle-up")
                }
                
                for key, (label, color, marker) in metrics.items():
                    h = client.get_metric_history(run_id, key)
                    if h:
                        steps = [m.step for m in h]
                        vals = [m.value for m in h]
                        fig_p.add_trace(go.Scatter(x=steps, y=vals, mode='lines+markers', name=label,
                                                 line=dict(color=color, width=2), marker=dict(size=8, symbol=marker)))

                fig_p.update_layout(title="GAN Training Progress (Epoch Averages)",
                                   xaxis_title="Epochs", yaxis_title="Average Loss",
                                   hovermode="x unified",
                                   font=dict(color="white"),
                                   paper_bgcolor='#1e293b',
                                   plot_bgcolor='#1e293b',
                                   xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                                   yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                fig_p.write_html(os.path.join(OUTPUT_DIR, "logs_gan_epoch.html"))
                print("Saved logs_gan_epoch.html")
            except Exception as e:
                print(f"Error creating Plotly GAN epoch: {e}")

            # --- Plot C: Validation Performance ---
            try:
                h_rmse = client.get_metric_history(run_id, "val_rmse")
                h_acc = client.get_metric_history(run_id, "val_acc")
                
                # --- Plotly Validation ---
                try:
                    fig_p = go.Figure()

                    if h_rmse and h_acc:
                         # RMSE (Left Axis)
                        steps_rmse = [m.step for m in h_rmse]
                        vals_rmse = [m.value for m in h_rmse]
                        fig_p.add_trace(go.Scatter(x=steps_rmse, y=vals_rmse, mode='lines+markers', name='Val RMSE',
                                                 line=dict(color='cyan', width=2), marker=dict(size=8)))
                        
                        # ACC (Right Axis)
                        steps_acc = [m.step for m in h_acc]
                        vals_acc = [m.value for m in h_acc]
                        fig_p.add_trace(go.Scatter(x=steps_acc, y=vals_acc, mode='lines+markers', name='Val ACC',
                                                 line=dict(color='orange', width=2), marker=dict(size=8, symbol='square'),
                                                 yaxis='y2'))
                        
                        fig_p.update_layout(title="GAN Validation Performance (Unseen Years)",
                                           xaxis_title="Epochs",
                                           yaxis=dict(title="RMSE (mm/day)", titlefont=dict(color="cyan"), tickfont=dict(color="cyan"), 
                                                      showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                                           yaxis2=dict(title="Anomaly Correlation (ACC)", titlefont=dict(color="orange"), tickfont=dict(color="orange"),
                                                       overlaying='y', side='right', showgrid=False),
                                           hovermode="x unified",
                                           paper_bgcolor='#1e293b',
                                           plot_bgcolor='#1e293b',
                                           font=dict(color="white"),
                                           xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                    else:
                        # Placeholder if no data
                        fig_p.update_layout(
                            title="GAN Validation Performance",
                            annotations=[dict(
                                text="No Validation Data Available Yet<br>Run training with --val-years argument",
                                xref="paper", yref="paper",
                                showarrow=False, font=dict(size=20, color="white")
                            )],
                            paper_bgcolor='#1e293b',
                            plot_bgcolor='#1e293b',
                            font=dict(color="white"),
                            xaxis=dict(visible=False, showgrid=False),
                            yaxis=dict(visible=False, showgrid=False)
                        )

                    fig_p.write_html(os.path.join(OUTPUT_DIR, "logs_gan_validation.html"))
                    print("Saved logs_gan_validation.html")
                except Exception as e:
                    print(f"Error creating Plotly GAN validation: {e}")

                # Matplotlib fallback (Optional, kept for png compatibility if needed)
                # ... (skipped for brevity as we are moving to html)

            except Exception as e:
                print(f"Skipping validation plot setup: {e}")

if __name__ == "__main__":
    plot_experiment_logs()
