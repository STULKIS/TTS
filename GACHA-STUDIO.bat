@echo off
setlocal
title Gacha TTS Studio
cd /d "%~dp0"

REM ============================================================
REM  Gacha TTS Studio - launcher
REM  Finds the GPT-SoVITS install at D:\GSV or C:\GSV and runs
REM  tools\type_ui.py with its runtime python.
REM  This window is the server - keep it open while you use it.
REM ============================================================

set "GSV="
for %%D in (D C) do if exist "%%D:\GSV\go-webui.bat" set "GSV=%%D:\GSV"
if not defined GSV for %%D in (D C) do if exist "%%D:\GSV" for /d %%F in ("%%D:\GSV\*") do if exist "%%F\go-webui.bat" set "GSV=%%F"
if not defined GSV (
    echo  [fail] GPT-SoVITS not found at D:\GSV or C:\GSV
    echo  Run INSTALL.bat first, then run me again.
    pause
    exit /b 1
)

echo  ============================================
echo   Gacha TTS Studio
echo   engine: %GSV%
echo  ============================================
echo.
echo  First load takes a few minutes - the model warms up.
echo  The browser opens at http://127.0.0.1:7861 by itself.
echo  Keep this window open while you use the Studio.
echo.
"%GSV%\runtime\python.exe" "%~dp0tools\type_ui.py" --gsv-root "%GSV%" --seeds "%~dp0seeds" --presets "%~dp0presets.csv" --open
echo.
echo  Studio closed.
pause
