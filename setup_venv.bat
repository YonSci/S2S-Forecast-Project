@echo off
setlocal

REM Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH. Please install Python first.
    exit /b 1
)

REM Create Virtual Environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo Virtual environment (.venv) already exists.
)

REM Activate Virtual Environment
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install Dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    echo NOTE: For geospatial libraries like Cartopy or Rioxarray on Windows, 
    echo you might need to install pre-compiled wheels (GDAL, etc.) manually 
    echo if the automatic install fails.
    exit /b 1
)

echo.
echo [SUCCESS] Virtual environment setup complete!
echo To activate the environment in the future, run:
echo    .venv\Scripts\activate.bat
echo.

echo IMPORTANT NOTE ABOUT CDO (Climate Data Operators):
echo --------------------------------------------------
echo The 'cdo' tool cannot be installed via pip.
echo You must install it manually on your system:
echo 1. Use WSL (Windows Subsystem for Linux) and install cdo there (recommended).
echo    Run in WSL: sudo apt-get install cdo
echo 2. Download the Windows binary from MPIMET.
echo    https://code.mpimet.mpg.de/projects/cdo/files
pause
