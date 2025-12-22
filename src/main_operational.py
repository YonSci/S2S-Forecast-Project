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
    data_dir = os.path.join(base_dir, "data", "raw")
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
    
    # 3. Run Inference
    logging.info(f"Step 3: Running inference on {pl_files[0]}...")
    try:
        run_inference(model_path, normalizer_path, pl_file, sst_file, output_dir=output_dir)
        logging.info("Inference successful. Maps generated in 'outputs/' folder.")
        
        # 4. Copy to fixed names for the web dashboard (latest_*.png)
        import shutil
        mapping = {
            'tercile': 'latest_tercile.png',
            'anomaly': 'latest_anomaly.png',
            'total': 'latest_total.png',
            'percent': 'latest_percent.png'
        }
        
        # Find the most recent files generated today/now
        all_outputs = os.listdir(output_dir)
        for key, target in mapping.items():
            matches = sorted([f for f in all_outputs if key in f and f.endswith('.png')], reverse=True)
            if matches:
                shutil.copy2(os.path.join(output_dir, matches[0]), os.path.join(output_dir, target))
                logging.info(f"Updated {target} with {matches[0]}")
                
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return

    logging.info("=== ET-NeuralCast Operational Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
