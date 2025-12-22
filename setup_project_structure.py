import os
import json
from pathlib import Path

def create_structure():
    # Define the project name with absolute path
    documents_path = os.path.expanduser("~/Documents")
    project_name = os.path.join(documents_path, "S2S-Forecast-Project")
    
    # Define the directory structure
    directories = [
        f"{project_name}/config",
        f"{project_name}/data/raw",
        f"{project_name}/data/processed",
        f"{project_name}/data/external",
        f"{project_name}/data/predictions",
        f"{project_name}/notebooks",
        f"{project_name}/src/data",
        f"{project_name}/src/models",
        f"{project_name}/src/training",
        f"{project_name}/src/utils",
        f"{project_name}/deploy",
        f"{project_name}/tests",
    ]
    
    # Define files to create with some initial content
    files = {
        f"{project_name}/config/data_config.yaml": "# Configuration for data loading, time ranges, and variable selection\n",
        f"{project_name}/config/model_config.yaml": "# Hyperparameters for the U-Net model (layers, filters, learning rate)\n",
        
        # Create valid empty notebooks with minimal JSON structure
        f"{project_name}/notebooks/01_data_inspection.ipynb": json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        f"{project_name}/notebooks/02_model_prototype.ipynb": json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        f"{project_name}/notebooks/03_error_analysis.ipynb": json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        
        # Source code files
        f"{project_name}/src/__init__.py": "",
        f"{project_name}/src/data/__init__.py": "",
        f"{project_name}/src/data/downloader.py": "# Scripts to fetch data from CDS (ERA5) and CHIRPS\n",
        f"{project_name}/src/data/preprocessor.py": "# Regridding, Normalization, and conversion to Tensors\n",
        f"{project_name}/src/data/dataloader.py": "# PyTorch/TF Dataset classes for efficient loading\n",
        
        f"{project_name}/src/models/__init__.py": "",
        f"{project_name}/src/models/components.py": "# Neural network building blocks (ConvBlocks, Attention, etc.)\n",
        f"{project_name}/src/models/unet.py": "# The main U-Net architecture definition\n",
        
        f"{project_name}/src/training/__init__.py": "",
        f"{project_name}/src/training/train.py": "# Main training loop logic\n",
        f"{project_name}/src/training/loss.py": "# Custom loss functions (e.g., weighted MSE for spatial focus)\n",
        f"{project_name}/src/training/metrics.py": "# Skill scores (ACC, RMSE) calculation\n",
        
        f"{project_name}/src/utils/__init__.py": "",
        
        # Deployment files
        f"{project_name}/deploy/Dockerfile": "# Container definition for reproducible environments\nFROM python:3.10-slim\n",
        f"{project_name}/deploy/app.py": "# FastAPI wrapper for model inference\n",
        
        # Root files
        f"{project_name}/env.yml": "name: s2s-forecast\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.10\n  - numpy\n  - xarray\n  - dask\n  - pytorch\n",
        f"{project_name}/requirements.txt": "numpy\npandas\nxarray\ndask\nnetCDF4\ntorch\ntorchvision\nrioxarray\nfastapi\nuvicorn\n",
        f"{project_name}/README.md": "# S2S Forecasting Project\n\nSpatial Deep Learning for Sub-seasonal to Seasonal Precipitation Forecasting.\n",
        f"{project_name}/.gitignore": "# Ignore large data files\ndata/\n\n# Python cache\n__pycache__/\n*.pyc\n\n# Jupyter checkpoints\n*.ipynb_checkpoints/\n\n# Environment vars\n.env\n.venv/\n",
    }

    print(f"Creating project structure for '{project_name}'...")

    # Create directories
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")

    # Create files
    for file_path, content in files.items():
        try:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Created file: {file_path}")
        except OSError as e:
            print(f"Error creating file {file_path}: {e}")

    print(f"\nProject structure created successfully in '{os.getcwd()}/{project_name}'")
    print("Don't forget to navigate into the directory:\n")
    print(f"    cd {project_name}")

if __name__ == "__main__":
    create_structure()
