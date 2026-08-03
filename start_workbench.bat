@echo off
rem Workbench launcher
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found.
    pause
    exit /b 1
)
start "" venv\Scripts\python.exe workbench.py