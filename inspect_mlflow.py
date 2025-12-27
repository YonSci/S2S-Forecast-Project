import mlflow
import os

DB_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(DB_URI)
client = mlflow.tracking.MlflowClient()

experiments = ["ET-NeuralCast-Warmstart", "ET-NeuralCast-GAN"]

print("=== MLflow Inspection ===")

for exp_name in experiments:
    print(f"\nExperiment: {exp_name}")
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        print("  Not found.")
        continue
        
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["attribute.start_time DESC"])
    if runs.empty:
        print("  No runs found.")
        continue
        
    # Pick the latest run
    run = runs.iloc[0]
    run_id = run.run_id
    print(f"  Latest Run ID: {run_id}")
    
    # List Metrics
    sim_run = client.get_run(run_id)
    metrics = sim_run.data.metrics.keys()
    print(f"  Available Metrics (for plotting):")
    for m in sorted(metrics):
        print(f"    - {m}")
        
    # List Artifacts
    artifacts = client.list_artifacts(run_id)
    print(f"  Available Artifacts (images/files):")
    if not artifacts:
        print("    None")
    for art in artifacts:
        print(f"    - {art.path} (Size: {art.file_size} bytes)")
        if art.is_dir:
            # list sub-artifacts if any
            sub_arts = client.list_artifacts(run_id, art.path)
            for s in sub_arts:
                 print(f"      - {s.path}")
