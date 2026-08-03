@echo off
rem Create a desktop shortcut named "manga-fanyi-zhushou" pointing to the packaged exe
cd /d "%~dp0"
set "EXE="
for /f "delims=" %%i in ('dir /b *.exe 2^>nul') do if not defined EXE set "EXE=%%~fi"
if not defined EXE (
    echo [ERROR] No .exe found in this folder.
    pause
    exit /b 1
)
echo Using: %EXE%
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws=New-Object -ComObject WScript.Shell; $n=[string][char]0x6F2B+[char]0x753B+[char]0x7FFB+[char]0x8BD1+[char]0x52A9+[char]0x624B; $lnk=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\'+$n+'.lnk'); $lnk.TargetPath='%EXE%'; $lnk.WorkingDirectory='%~dp0'; $lnk.IconLocation='%~dp0app.ico'; $lnk.Save()"
if %errorlevel%==0 (
    echo Shortcut created on Desktop.
) else (
    echo Failed. Create manually: right-click exe - Send to - Desktop shortcut.
)
pause
