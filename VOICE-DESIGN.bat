@echo off
setlocal
title Gacha Voice Designer - CPU
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_voice_design.ps1"
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo Setup or launch failed. The error is above. Classic Studio was not changed.
pause
exit /b %RESULT%
