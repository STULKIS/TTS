@echo off
setlocal EnableDelayedExpansion
title Gacha TTS - repair
cd /d "%~dp0"

REM Re-applies CPU mode and runs the pre-flight checks.
REM Run this whenever something stops working - then send the output if it fails.

set "GSV_ROOT=D:\GSV"

echo.
echo  ============================================
echo   Gacha TTS - repair
echo  ============================================
echo.

if exist "%GSV_ROOT%\go-webui.bat" goto :gsv_ok
if exist "%GSV_ROOT%" for /d %%F in ("%GSV_ROOT%\*") do if exist "%%~F\go-webui.bat" set "GSV_ROOT=%%~F"
if exist "%GSV_ROOT%\go-webui.bat" goto :gsv_ok
echo  [fail] GPT-SoVITS not found at D:\GSV
echo  Run INSTALL.bat first (it unpacks the engine), then run me again.
pause
exit /b 1
:gsv_ok
echo  [ok] GPT-SoVITS found at %GSV_ROOT%

set "PYCMD="
py -3 --version >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD python --version >nul 2>nul && set "PYCMD=python"
if not defined PYCMD (
    echo  [fail] no Python found on PATH.
    echo  Launch the WebUI once from the desktop icon first (it uses the
    echo  Python bundled with the package), then run me again.
    pause
    exit /b 1
)

!PYCMD! "%~dp0tools\cpu_config.py" --gsv-root "%GSV_ROOT%"
!PYCMD! "%~dp0tools\preflight.py" --gsv-root "%GSV_ROOT%" --seeds "%~dp0seeds"

echo.
echo  If everything above says [ok], launch from the "Gacha TTS" desktop icon.
echo  If something failed, copy this window's text and send it for help.
pause
