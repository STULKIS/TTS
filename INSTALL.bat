@echo off
setlocal EnableDelayedExpansion
title Gacha TTS - one-click installer (CPU mode)
cd /d "%~dp0"

REM ============================================================
REM  Gacha TTS - one-click installer (no-NVIDIA / CPU route)
REM  Double-click it. It will:
REM    0. let you CHOOSE the install drive
REM    1. put this repo at <drive>:\TTS
REM    2. unpack the GPT-SoVITS v2pro package to <drive>:\GSV
REM       the ~8 GB .7z is a one-time download, not bundled
REM    3. switch GPT-SoVITS to CPU mode for no-NVIDIA boxes
REM    4. create a "Gacha TTS" desktop icon
REM    5. open the type-a-text WebUI
REM  The window ALWAYS stays open at the end and shows any error.
REM  NOTE: no parentheses allowed in echo lines inside blocks -
REM  cmd parses them as code.
REM ============================================================

REM Never let this window close silently: everything runs as a
REM subroutine, and the tail always pauses and shows the exit code.
call :main
if defined CLOSED exit /b 0
echo.
echo  ----------------------------------------
echo  Installer finished. Exit code: %ERRORLEVEL%
echo  If something above looks wrong, copy this
echo  whole window's text and send it for help.
echo  ----------------------------------------
echo.
pause
exit /b %ERRORLEVEL%

:main
echo.
echo  ===============================================
echo   Gacha TTS - one-click installer, CPU mode
echo   repo folder: %~dp0
echo  ===============================================
echo.

REM ---------- 0) choose the install drive ----------
echo  [0/5] choose the install drive
echo  I need about 10 GB free: the TTS folder + the GPT-SoVITS engine.
echo.
for /f "tokens=1,2" %%a in ('powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem | ForEach-Object { '{0} {1:N0}' -f $_.Name, ($_.Free/1GB) }"') do echo    %%a:   %%2 GB free
echo.
set "DRV="
set /p "DRV=Type the drive letter to install on, empty means D if D exists: "
set "DRV=!DRV:~0,1!"
if "!DRV!"=="" (
    if exist "D:\" set "DRV=D" else set "DRV=C"
)
if not exist "!DRV!:\" (
    echo  No such drive: !DRV!
    exit /b 1
)
set "TTS_ROOT=!DRV!:\TTS"
set "GSV_ROOT=!DRV!:\GSV"
echo.
echo  Installing to:  !TTS_ROOT!  +  !GSV_ROOT!
echo.
timeout /t 3 >nul

REM ---------- 1) put this repo at TTS_ROOT ----------
if /i not "%cd%"=="%TTS_ROOT%" (
    echo  [1/5] this repo is currently at:  %cd%
    echo        the installer wants it at:   %TTS_ROOT%
    echo.
    set /p "GOA=Copy it there now? [Y/n] "
    if /i "!GOA!"=="n" (
        echo  OK - copy the folder to %TTS_ROOT% by hand,
        echo  then run INSTALL.bat again from there.
        exit /b 1
    )
    echo  Copying now, can take a couple of minutes...
    xcopy /E /I /Y /Q "%cd%" "%TTS_ROOT%" >nul
    if errorlevel 1 (
        echo  Copy failed. Close any Explorer window on this folder and re-run.
        exit /b 1
    )
    if exist "%TTS_ROOT%\INSTALL.bat" (
        echo  Re-launching from %TTS_ROOT% ...
        start "" "%TTS_ROOT%\INSTALL.bat"
        set "CLOSED=1"
        exit /b 0
    )
    echo  Copy partially failed - re-run me.
    exit /b 1
)
echo  [1/5] repo OK at %cd%

REM ---------- 2) GPT-SoVITS engine ----------
if exist "%GSV_ROOT%\go-webui.bat" goto :gsv_ok
if exist "%GSV_ROOT%" for /d %%F in ("%GSV_ROOT%\*") do if exist "%%~F\go-webui.bat" set "GSV_ROOT=%%~F"
if exist "%GSV_ROOT%\go-webui.bat" goto :gsv_ok

echo  [2/5] GPT-SoVITS not found at %GSV_ROOT%
echo.
echo  You need the ~8 GB package  GPT-SoVITS-v2pro-20250604.7z
echo  one-time download from Hugging Face, link below.
echo.
set /p "SEVENZ=  Where is the .7z file? full path, or N if not downloaded yet: "
if /i "!SEVENZ!"=="" goto :need_download
if /i "!SEVENZ!"=="N" goto :need_download
if /i "!SEVENZ!"=="n" goto :need_download
if not exist "!SEVENZ!" (
    echo  Not found:  !SEVENZ!
    exit /b 1
)

set "SEVENZIP="
if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles%\7-Zip\7z.exe"
if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"
where 7z.exe >nul 2>nul && set "SEVENZIP=7z.exe"
if not defined SEVENZIP (
    echo.
    echo  7-Zip is needed to unpack the package.
    echo  Install it from https://7-zip.org , then re-run this installer.
    exit /b 1
)

echo.
echo  Unpacking !SEVENZ! to %GSV_ROOT%
echo  8 GB takes a while, 10-30 min on an SSD. Go get a drink.
echo  Do not close this window.
echo.
mkdir "%GSV_ROOT%" 2>nul
!SEVENZIP! x "!SEVENZ!" -o"%GSV_ROOT%" -y >"%TTS_ROOT%\unpack.log" 2>&1
if errorlevel 1 (
    echo  Unpack failed - details in %TTS_ROOT%\unpack.log
    exit /b 1
)
if exist "%GSV_ROOT%" for /d %%F in ("%GSV_ROOT%\*") do if exist "%%~F\go-webui.bat" set "GSV_ROOT=%%~F"
if not exist "%GSV_ROOT%\go-webui.bat" (
    echo  Unpacked, but go-webui.bat was not found under %GSV_ROOT% .
    echo  Check %TTS_ROOT%\unpack.log and re-run.
    exit /b 1
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
    echo  Opening the WebUI - on the 0-Automatic TTS tab, type text and
    echo  pick a reference clip, 5 to 10 seconds long, to choose the voice.
    start "" "%GSV_ROOT%\go-webui.bat"
)
echo.
echo  Re-launch any time from the "Gacha TTS" desktop icon.
echo  First load is slow - the model needs a couple of minutes to warm up.
exit /b 0

:need_download
echo.
echo  Get the package first, one time, ~8 GB:
echo    https://huggingface.co/lj1995/GPT-SoVITS-windows-package/tree/main
echo  pick:  GPT-SoVITS-v2pro-20250604.7z
echo  if that exact file is gone, pick the newest GPT-SoVITS-v2pro file.
echo.
echo  Save it anywhere on a big drive, and re-run this installer.
exit /b 1
