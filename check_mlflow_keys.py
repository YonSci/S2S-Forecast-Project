import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

run_id = "4ee1594fd34c43bb888d89478482c167" # Warmstart best run
print(f"Inspecting Run: {run_id}")
run = client.get_run(run_id)

print("Metrics keys:", run.data.metrics.keys())

# Check history for one valid key
if run.data.metrics:
    first_key = list(run.data.metrics.keys())[0]
    history = client.get_metric_history(run_id, first_key)
    print(f"History for {first_key}: {len(history)} entries")
else:
    print("No metrics found.")
