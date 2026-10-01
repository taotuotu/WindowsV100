@echo off
setlocal
pushd "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop-ninfer.ps1" %*
set "exit_code=%ERRORLEVEL%"
if not "%exit_code%"=="0" pause
popd
endlocal & exit /b %exit_code%
