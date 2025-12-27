@echo off
echo Starting MLflow UI...
echo Access the dashboard at http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
pause
