import os
import json
import glob
import re

OUTPUT_DIR = "outputs"

def update_index():
    print("Updating forecast_index.json based on existing files...")
    
    # Pattern: forecast_YYYY-MM-DD_WX_anomaly.png
    # We only care about unique DATE and LEAD combinations.
    files = glob.glob(os.path.join(OUTPUT_DIR, "forecast_*_*_anomaly.png"))
    
    leads = {}
    
    for f in files:
        basename = os.path.basename(f)
        # extract date and lead
        # forecast_2025-01-12_W1_anomaly.png
        match = re.match(r"forecast_(\d{4}-\d{2}-\d{2})_(W\d)_anomaly\.png", basename)
        if match:
            date = match.group(1)
            lead = match.group(2)
            
            if lead not in leads:
                leads[lead] = []
            
            if date not in leads[lead]:
                leads[lead].append(date)
    
    # Sort dates descending
    for lead in leads:
        leads[lead] = sorted(leads[lead], reverse=True)
        
    index = {
        "last_updated": "now",
        "leads": leads
    }
    
    with open(os.path.join(OUTPUT_DIR, "forecast_index.json"), "w") as f:
        json.dump(index, f, indent=4)
        
    print(f"Index updated: {leads}")

if __name__ == "__main__":
    update_index()
