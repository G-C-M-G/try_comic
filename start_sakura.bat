@echo off
rem Sakura local translation service launcher (127.0.0.1:8080)
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Please setup environment first.
    pause
    exit /b 1
)

rem Check if service is already running
curl -s -m 2 http://127.0.0.1:8080/health >nul 2>&1
if %errorlevel%==0 (
    echo Sakura service is already running (http://127.0.0.1:8080)
    pause
    exit /b 0
)

echo Starting sakura service...
start "sakura-server" /min venv\Scripts\python.exe sakura\sakura_server.py
echo Started. Waiting for model loading (about 10-30s)...
timeout /t 20 /nobreak >nul

rem Check again
curl -s -m 2 http://127.0.0.1:8080/health >nul 2>&1
if %errorlevel%==0 (
    echo Service ready: http://127.0.0.1:8080
) else (
    echo Service may still be loading. Check sakura\server.err.log if it fails.
)
pause