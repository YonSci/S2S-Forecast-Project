import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.unet import UNetGenerator
from src.data.dataloader import S2SDataset
from src.data.normalization import S2SNormalizer
from src.evaluation.metrics import compute_metrics, mae, hit_rate
# Climatology logic is embedded in standardization; Persistence logic implemented inline
# from src.evaluation.baselines import ClimatologyBaseline 

def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    
    # 1. Load Normalizer
    normalizer = S2SNormalizer()
    if not os.path.exists(config['normalizer_path']):
        raise FileNotFoundError(f"Normalizer not found at {config['normalizer_path']}")
    normalizer.load(config['normalizer_path'])
    
    # 2. Test Dataset
    test_dataset = S2SDataset(
        data_dir=config['data_dir'],
        years=config['test_years'],
        mode='test',
        normalizer=normalizer,
        lead_weeks=config['lead_weeks']
    )
    
    if len(test_dataset) == 0:
        print("No samples in test dataset!")
        return

    # 3. Model
    model = UNetGenerator(input_channels=5, output_channels=1, target_size=(48, 60)).to(device)
    try:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {config['model_path']}")
        return
        
    model.eval()
    
    # 4. Tercile Edges (Approximate for Normalized Data N(0,1))
    # 33rd percentile ~ -0.43, 67th percentile ~ +0.43
    TERCILE_EDGES = [-0.43, 0.43]
    
    results = []
    
    print(f"Running evaluation on {len(test_dataset)} samples...")
    print(f"Using Tercile Edges for Normalized Data: {TERCILE_EDGES}")
    
    # Collect all targets first for Persistence (Lagged Prediction)
    # Limitation: This persistence logic only works if dataset is temporal and continuous
    # We will assume test_dataset indices imply time order.
    # Persistence Pred for Sample i = Target of Sample (i - Lead_Weeks)
    # Ideally: Persistence Pred for Week T+Lead = Observation at Week T
    # Sample i: Input(Week T), Target(Week T+Lead)
    # We want Observation at Week T. 
    # Can we get that? 
    # Yes, we can infer it if we had the "Target" of Sample (i-Lead)? 
    # Sample (i-Lead): Input(Week T-Lead), Target(Week T).
    # Correct. So Persistence is Target of Sample (i - Lead).
    
    all_targets = []
    # Pre-fetch all targets to handle persistence easily
    for i in range(len(test_dataset)):
        _, target = test_dataset[i]
        all_targets.append(target.numpy())
        
    all_targets = np.array(all_targets) # (N, H, W)

    for i in range(len(test_dataset)):
        # Inputs
        input_tensor, target_tensor = test_dataset[i]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        target_np = target_tensor.numpy()
        
        # A. Model Predict
        with torch.no_grad():
            pred_tensor = model(input_tensor).cpu().numpy()[0]
            
        # B. Baselines
        # 1. Climatology (Normalized) -> All Zeros
        clim_pred = np.zeros_like(target_np)
        
        # 2. Persistence
        # Pred for sample i (Week T+Lead) uses Obs from Week T.
        # Week T is found at target index (i - lead_weeks)
        # Because Sample j has Target j+Lead.
        # We want Target T.
        # Sample (i-Lead) has Target (i-Lead+Lead) = Target i. NO.
        # Sample s has Target Week (s + Lead).
        # We need Week T.
        # Sample i corresponds to Input Week i.
        # We want Target of Week i.
        # Week i is Target of Sample (i - Lead).
        # Logic: Sample (i-Lead) -> Input (i-Lead), Target (i-Lead+Lead) = Target (i).
        # So yes, Persistence Pred = Target of Sample (i - Lead_Weeks).
        
        pidx = i - config['lead_weeks']
        if pidx >= 0:
            pers_pred = all_targets[pidx]
        else:
            # Cannot align (start of series), use Climatology fallback
            pers_pred = np.zeros_like(target_np)
            
            
        # Compute Metrics for ALL models
        def get_scores(prediction, truth, label):
            metrics = compute_metrics(prediction, truth, 0.0, TERCILE_EDGES) # Climatology is 0
            metrics['MAE'] = mae(prediction, truth)
            
            # Additional Hit Rate calculation
            # (already covered by HSS logic roughly, but let's be explicit)
            # Re-categorize locally
            def cat(d):
                c = np.zeros_like(d, dtype=int)
                c[d < TERCILE_EDGES[0]] = -1
                c[d > TERCILE_EDGES[1]] = 1
                return c
            
            p_cat = cat(prediction)
            t_cat = cat(truth)
            metrics['HitRate'] = hit_rate(p_cat, t_cat)
            
            # Rename keys
            return {f"{k}_{label}": v for k, v in metrics.items()}

        scores_model = get_scores(pred_tensor, target_np, "Model")
        scores_clim = get_scores(clim_pred, target_np, "Clim")
        scores_pers = get_scores(pers_pred, target_np, "Pers")
        
        # Merge dicts
        row = {**scores_model, **scores_clim, **scores_pers}
        results.append(row)

    # Aggregation
    df = pd.DataFrame(results)
    # Drop first few rows where Persistence was invalid
    df = df.iloc[config['lead_weeks']:]
    
    print("\n" + "="*40)
    print(f" EVALUATION REPORT (Lead: {config['lead_weeks']} Weeks)")
    print(f" Test Years: {config['test_years']}")
    print(f" Samples Evaluated: {len(df)}")
    print("="*40)
    
    headers = ["Metric", "Model", "Persistence", "Climatology", "Skill (vs Clim)"]
    print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<15}")
    print("-"*66)
    
    metrics_to_show = ["RMSE", "MAE", "ACC", "HSS", "HitRate"]
    
    for m in metrics_to_show:
        val_mod = df[f"{m}_Model"].mean()
        val_pers = df[f"{m}_Pers"].mean()
        val_clim = df[f"{m}_Clim"].mean()
        
        # Skill Score Calculation (Standard: 1 - MSE_mod/MSE_clim)
        # For correlation/accuracy: (Mod - Clim)/(Perf - Clim) -> Different
        # Simple Logic: How much better than Clim?
        if m in ["RMSE", "MAE"]:
            # Lower is better. Skill = 1 - (Mod / Clim)
            skill = 1 - (val_mod / val_clim)
        else:
            # Higher is better. Skill = Mod - Clim (Simple difference)
            skill = val_mod - val_clim
            
        print(f"{m:<15} {val_mod:<12.4f} {val_pers:<12.4f} {val_clim:<12.4f} {skill:<12.4f}")
        
    print("="*40)
    print("Notes:")
    print("- Skill for Error Metrics (RMSE, MAE): Positive is Good (Reduction in error).")
    print("- Skill for Score Metrics (ACC, HSS): Positive is Good (Increase in score).")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, nargs='+', default=["2008", "2009"], help="Test years")
    parser.add_argument("--lead", type=int, default=1, help="Lead weeks")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    # Parse years
    years = []
    for y in args.years:
        if '-' in y:
            s, e = map(int, y.split('-'))
            years.extend(range(s, e+1))
        else:
            years.append(int(y))
            
    config = {
        'data_dir': 'data/train',
        'test_years': years,
        'lead_weeks': args.lead,
        'model_path': args.model,
        'normalizer_path': 'checkpoints/normalizer.pkl'
    }
    
    evaluate_model(config)
