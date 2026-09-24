@echo off
setlocal EnableDelayedExpansion
title Gacha TTS - one-click installer (CPU mode)
cd /d "%~dp0"

REM ============================================================
REM  Gacha TTS - one-click installer (no-NVIDIA / CPU route)
REM  Double-click it. It will:
REM    1. put this repo at D:\TTS
REM    2. unpack the GPT-SoVITS v2pro package to D:\GSV
REM       (the ~8 GB .7z is a one-time download, not bundled)
REM    3. switch GPT-SoVITS to CPU mode (no NVIDIA)
REM    4. create a "Gacha TTS" desktop icon
REM    5. open the type-a-text WebUI
REM
REM  No D: drive? Edit TTS_ROOT / GSV_ROOT below.
REM ============================================================

set "TTS_ROOT=D:\TTS"
set "GSV_ROOT=D:\GSV"

echo.
echo  ===============================================
echo   Gacha TTS - one-click installer (CPU mode)
echo  ===============================================
echo.

REM ---------- 1) put this repo at TTS_ROOT ----------
if /i not "%cd%"=="%TTS_ROOT%" (
    echo  [1/5] this repo is currently at:  %cd%
    echo        the installer wants it at:   %TTS_ROOT%
    echo.
    set /p "GOA=Copy it there now? [Y/n] "
    if /i "!GOA!"=="n" goto :abort
    echo Copying now (can take a couple of minutes)...
    xcopy /E /I /Y /Q "%cd%" "%TTS_ROOT%" >nul
    if errorlevel 1 (
        echo Copy failed. Close any Explorer window on this folder and re-run.
        goto :abort
    )
    echo Re-launching from %TTS_ROOT% ...
    start "" "%TTS_ROOT%\INSTALL.bat"
    exit /b 0
)
echo  [1/5] repo OK at %cd%

REM ---------- 2) GPT-SoVITS engine ----------
if exist "%GSV_ROOT%\go-webui.bat" goto :gsv_ok
if exist "%GSV_ROOT%" for /d %%F in ("%GSV_ROOT%\*") do if exist "%%~F\go-webui.bat" set "GSV_ROOT=%%~F"
if exist "%GSV_ROOT%\go-webui.bat" goto :gsv_ok

echo  [2/5] GPT-SoVITS not found at %GSV_ROOT%
echo.
echo  You need the ~8 GB package  GPT-SoVITS-v2pro-20250604.7z
echo  (one-time download from Hugging Face - link below).
echo.
set /p "SEVENZ=  Where is the .7z file? (full path, or N if not downloaded yet) "
if /i "!SEVENZ!"=="" goto :need_download
if /i "!SEVENZ!"=="N" goto :need_download
if /i "!SEVENZ!"=="n" goto :need_download
if not exist "!SEVENZ!" (
    echo  Not found:  !SEVENZ!
    goto :abort
)

set "SEVENZIP="
if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles%\7-Zip\7z.exe"
if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"
where 7z.exe >nul 2>nul && set "SEVENZIP=7z.exe"
if not defined SEVENZIP (
    echo.
    echo  7-Zip is needed to unpack the package.
    echo  Install it from https://7-zip.org , then re-run this installer.
    goto :abort
)

echo.
echo  Unpacking !SEVENZ! to %GSV_ROOT%
echo  8 GB takes a while (10-30 min on an SSD). Go get a drink.
echo.
mkdir "%GSV_ROOT%" 2>nul
!SEVENZIP! x "!SEVENZ!" -o"%GSV_ROOT%" -y >"%TTS_ROOT%\unpack.log" 2>&1
if errorlevel 1 (
    echo  Unpack failed - details in %TTS_ROOT%\unpack.log
    goto :abort
)
if exist "%GSV_ROOT%" for /d %%F in ("%GSV_ROOT%\*") do if exist "%%~F\go-webui.bat" set "GSV_ROOT=%%~F"
if not exist "%GSV_ROOT%\go-webui.bat" (
    echo  Unpacked, but go-webui.bat was not found under %GSV_ROOT% .
    echo  Check %TTS_ROOT%\unpack.log and re-run.
    goto :abort
)
echo  Unpacked OK.
:gsv_ok
echo  [2/5] GPT-SoVITS OK at %GSV_ROOT%

REM ---------- 3) CPU mode (no NVIDIA) ----------
echo  [3/5] CPU mode
set "PYCMD="
py -3 --version >nul 2>nul && set "PYCMD=py -3"
if not defined PYCMD python --version >nul 2>nul && set "PYCMD=python"
if defined PYCMD (
    !PYCMD! "%TTS_ROOT%\tools\cpu_config.py" --gsv-root "%GSV_ROOT%"
    !PYCMD! "%TTS_ROOT%\tools\preflight.py" --gsv-root "%GSV_ROOT%" --seeds "%TTS_ROOT%\seeds"
) else (
    echo  No Python on PATH - applying the CPU config edit directly...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$f='%GSV_ROOT%\GPT_SoVITS\configs\tts_infer.yaml'; if (Test-Path $f) { $t=[IO.File]::ReadAllText($f); $t=$t -replace '(?m)^(\s*)device:\s*\S+','${1}device: cpu'; $t=$t -replace '(?m)^(\s*)is_half:\s*true\b','${1}is_half: false'; [IO.File]::WriteAllText($f,$t); echo '  [ok] tts_infer.yaml set to CPU mode' } else { echo '  [warn] tts_infer.yaml not found at expected path' }"
)

REM ---------- 4) desktop icon ----------
echo  [4/5] desktop icon
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws=New-Object -ComObject WScript.Shell; $d=[Environment]::GetFolderPath('Desktop'); $l=$ws.CreateShortcut($d+'\Gacha TTS.lnk'); $l.TargetPath='%GSV_ROOT%\go-webui.bat'; $l.WorkingDirectory='%GSV_ROOT%'; $l.Description='Gacha TTS - type text, hear your characters'; $l.IconLocation='shell32.dll,13'; $l.Save(); echo '  [ok] desktop icon created'"
if errorlevel 1 echo  [warn] desktop icon could not be created

REM ---------- 5) launch ----------
echo  [5/5] all done
echo.
set /p "RUN=Launch Gacha TTS now? [Y/n] "
if /i not "!RUN!"=="n" (
    echo Opening the WebUI - on the 0-Automatic TTS tab, type text and
    echo pick a reference clip (5-10 s) to choose the voice.
    start "" "%GSV_ROOT%\go-webui.bat"
)
echo.
echo  Re-launch any time from the "Gacha TTS" desktop icon.
echo  First load is slow - the model needs a couple of minutes to warm up.
echo.
pause
exit /b 0

:need_download
echo.
echo  Get the package first (one time, ~8 GB):
echo    https://huggingface.co/lj1995/GPT-SoVITS-windows-package/tree/main
echo  pick:  GPT-SoVITS-v2pro-20250604.7z
echo  (if that exact file is gone, pick the newest GPT-SoVITS-v2pro-*.7z)
echo.
echo  Save it anywhere you like (e.g. D:\downloads\) and re-run this installer.
echo.
pause
exit /b 1

:abort
echo.
echo  Stopped. Nothing was broken - re-run me when ready.
echo.
pause
exit /b 1
