@echo off
setlocal EnableDelayedExpansion
title Gacha TTS - uninstall
cd /d "%~dp0"

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

REM ---------- GPT-SoVITS engine, D: then C: ----------
for %%D in (D C) do call :rm_gsv %%D
REM ---------- repo copy, D: then C: ----------
for %%D in (D C) do call :rm_tts %%D

echo.
echo  Done. If you keep the .7z package somewhere, you can reinstall
echo  any time by double-clicking INSTALL.bat again.
echo.
pause
exit /b 0

:rm_gsv
set "P=%~1:\GSV"
set /p "A=Remove %P% , the 8 GB engine? [y/N] "
if /i "!A!"=="y" (
    if exist "%P%" (
        echo  Removing %P% ...
        echo  Close the WebUI first if it is open, or files will be locked.
        rmdir /s /q "%P%" 2>nul
        if exist "%P%" (echo  [warn] still there - close the WebUI and re-run) else (echo  [ok] removed)
    ) else (echo  [ok] not found, nothing to remove)
)
goto :eof

:rm_tts
set "P=%~1:\TTS"
set /p "A=Remove %P% , the TTS folder? [y/N] "
if /i "!A!"=="y" (
    if /i "%cd%"=="%P%" (
        echo  [skip] this window is running from inside that folder -
        echo  after it closes, delete %P% by hand
    ) else if exist "%P%" (
        echo  Removing %P% ...
        rmdir /s /q "%P%" 2>nul
        if exist "%P%" (echo  [warn] still there - close anything using it and re-run) else (echo  [ok] removed)
    ) else (echo  [ok] not found, nothing to remove)
)
goto :eof
