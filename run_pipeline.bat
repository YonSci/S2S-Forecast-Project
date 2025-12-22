@echo off
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo [!] Starting ET-NeuralCast Operational Pipeline...
.\s2s_env\Scripts\python.exe src\main_operational.py

if %ERRORLEVEL% NEQ 0 (
    echo [X] Pipeline failed! Check pipeline.log for details.
    pause
    exit /b %ERRORLEVEL%
)

echo [+] Pipeline execution finished successfully.
echo [*] Maps are available in the 'outputs/' directory.
pause
