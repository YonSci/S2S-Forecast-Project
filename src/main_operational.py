import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.download_operational import download_operational_data
from src.inference.predict import run_inference

# Configure Logging
log_file = "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    logging.info("=== ET-NeuralCast Operational Pipeline Started ===")
    
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data", "operational")
    pl_dir = os.path.join(data_dir, "era5_pressure")
    sst_dir = os.path.join(data_dir, "era5_sst")
    model_path = os.path.join(base_dir, "checkpoints", "G_warmstart_best.pth")
    normalizer_path = os.path.join(base_dir, "checkpoints", "normalizer.pkl")
    output_dir = os.path.join(base_dir, "outputs")
    
    os.makedirs(pl_dir, exist_ok=True)
    os.makedirs(sst_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Download Latest Data
    logging.info("Step 1: Downloading operational data from CDS...")
    try:
        download_operational_data(pl_dir, sst_dir)
        logging.info("Download complete.")
    except Exception as e:
        logging.error(f"Download failed: {e}")
        # Note: We continue because we might already have data from a previous day
    
    # 2. Identify the newest file for inference
    logging.info("Step 2: Identifying latest data for inference...")
    pl_files = sorted([f for f in os.listdir(pl_dir) if f.endswith('.nc')], reverse=True)
    if not pl_files:
        logging.error("No ERA5 pressure files found. Pipeline aborted.")
        return
        
    pl_file = os.path.join(pl_dir, pl_files[0])
    sst_file = os.path.join(sst_dir, f"era5_sst_{pl_files[0].replace('era5_pressure_', '')}")
    if not os.path.exists(sst_file):
        sst_file = None
        logging.warning(f"SST file not found for {pl_files[0]}. Running without SST.")
    
    # 3. Run Inference for each Lead
    leads = [1, 2, 3, 4]
    for lead in leads:
        logging.info(f"Step 3: Running inference for LEAD WEEK {lead}...")
        lead_model_path = os.path.join(base_dir, "checkpoints", f"G_gan_epoch_30_W{lead}.pth")
        
        # Fallback to warmstart if GAN doesn't exist yet
        if not os.path.exists(lead_model_path):
            lead_model_path = os.path.join(base_dir, "checkpoints", f"G_warmstart_best_W{lead}.pth")
            
        if not os.path.exists(lead_model_path):
            logging.warning(f"No model found for lead W{lead} at {lead_model_path}. Skipping.")
            continue
            
        try:
            run_inference(lead_model_path, normalizer_path, pl_file, sst_file, output_dir=output_dir, lead_weeks=lead)
            logging.info(f"Inference successful for Lead W{lead}.")
        except Exception as e:
            logging.error(f"Inference failed for Lead W{lead}: {e}")
            continue

    # 4. Copy to fixed names for the web dashboard (latest_*_W*.png)
    import shutil
    try:
        mapping_keys = ['tercile', 'anomaly', 'total', 'percent']
        all_outputs = os.listdir(output_dir)
        
        for lead in leads:
            for key in mapping_keys:
                target = f"latest_{key}_W{lead}.png"
                # Pattern: forecast_YYYY-MM-DD_W{lead}_{key}.png
                pattern = f"_W{lead}_{key}.png"
                matches = sorted([f for f in all_outputs if pattern in f], reverse=True)
                if matches:
                    shutil.copy2(os.path.join(output_dir, matches[0]), os.path.join(output_dir, target))
                    logging.info(f"Updated {target} with {matches[0]}")
    except Exception as e:
        logging.error(f"Copying latest files failed: {e}")

    # 5. Generate Manifest for Web Dashboard (forecast_index.json)
    try:
        import json
        import re
        
        forecast_data = {} # lead -> dates
        date_pattern = re.compile(r"forecast_(\d{4}-\d{2}-\d{2})_W(\d)")
        
        for f in os.listdir(output_dir):
            match = date_pattern.search(f)
            if match:
                date_str = match.group(1)
                lead_num = match.group(2)
                if lead_num not in forecast_data:
                    forecast_data[lead_num] = set()
                forecast_data[lead_num].add(date_str)
        
        # Format for JSON
        formatted_data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "leads": {
                f"W{l}": sorted(list(dates), reverse=True) for l, dates in forecast_data.items()
            }
        }
        
        manifest_path = os.path.join(output_dir, "forecast_index.json")
        with open(manifest_path, 'w') as f:
            json.dump(formatted_data, f, indent=4)
        logging.info(f"Generated multi-lead manifest: {manifest_path}")
    except Exception as e:
        logging.error(f"Manifest generation failed: {e}")
            
    except Exception as e:
        logging.error(f"Operational pipeline failed: {e}")
        return

    logging.info("=== ET-NeuralCast Operational Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
