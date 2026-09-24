@echo off
setlocal EnableDelayedExpansion
title Gacha TTS - uninstall
cd /d "%~dp0"

set "TTS_ROOT=D:\TTS"
set "GSV_ROOT=D:\GSV"

echo.
echo  ============================================
echo   Gacha TTS - uninstall
echo  ============================================
echo
echo  This removes the desktop icon and, if you say so,
echo  the installed folders. Your .7z download file is NOT touched.
echo.

REM ---------- desktop icon ----------
powershell -NoProfile -Command "$p=[Environment]::GetFolderPath('Desktop')+'\Gacha TTS.lnk'; if (Test-Path $p) { Remove-Item $p; echo '  [ok] desktop icon removed' } else { echo '  [ok] no desktop icon found' }"

REM ---------- GPT-SoVITS engine ----------
set /p "RGSV=Remove %GSV_ROOT% (the ~8 GB engine)? [y/N] "
if /i "!RGSV!"=="y" (
    if exist "%GSV_ROOT%" (
        echo  Removing %GSV_ROOT% ...
        echo  (close the WebUI first if it is open, or files will be locked)
        rmdir /s /q "%GSV_ROOT%" 2>nul
        if exist "%GSV_ROOT%" (echo  [warn] still there - close the WebUI and re-run) else (echo  [ok] removed)
    ) else (echo  [ok] not found, nothing to remove)
)

REM ---------- repo copy ----------
set /p "RTTS=Remove %TTS_ROOT% (the TTS folder)? [y/N] "
if /i "!RTTS!"=="y" (
    if /i "%cd%"=="%TTS_ROOT%" (
        echo  [skip] this window is running from inside that folder -
        echo  after it closes, delete %TTS_ROOT% by hand
    ) else if exist "%TTS_ROOT%" (
        echo  Removing %TTS_ROOT% ...
        rmdir /s /q "%TTS_ROOT%" 2>nul
        if exist "%TTS_ROOT%" (echo  [warn] still there - close anything using it and re-run) else (echo  [ok] removed)
    ) else (echo  [ok] not found, nothing to remove)
)

echo.
echo  Done. If you keep the .7z package somewhere, you can reinstall
echo  any time by double-clicking INSTALL.bat again.
echo.
pause
exit /b 0
