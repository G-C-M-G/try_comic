@echo off
rem 漫画翻译助手 - 统一入口（pythonw 无控制台）
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
    echo [ERROR] venv not found.
    pause
    exit /b 1
)
start "" venv\Scripts\pythonw.exe app.py
