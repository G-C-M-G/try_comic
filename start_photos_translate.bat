@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag one manga image onto this file, or pass its path as the first argument.
  pause
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" "%~dp0photos_translate.py" "%~1"
if errorlevel 1 pause
