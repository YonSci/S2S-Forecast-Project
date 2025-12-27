import os
import glob
import re
import json
import pandas as pd

def generate_manifest(output_dir):
    print(f"Generating manifest for {output_dir}...")
    
    # improved regex to find years in filenames like "spatial_bias_2020.png"
    # We look for files ending in _YYYY.png
    files = glob.glob(os.path.join(output_dir, "*_*.png"))
    years = set()
    
    for f in files:
        match = re.search(r'_(\d{4})\.png$', f)
        if match:
            years.add(int(match.group(1)))
            
    sorted_years = sorted(list(years))
    
    manifest = {
        "available_years": sorted_years,
        "latest_year": sorted_years[-1] if sorted_years else None,
        "generated_at": pd.Timestamp.now().isoformat()
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest saved to {manifest_path}: {sorted_years}")

if __name__ == "__main__":
    generate_manifest("outputs")
