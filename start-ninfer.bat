@echo off
setlocal
pushd "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-ninfer-server.ps1" %*
set "exit_code=%ERRORLEVEL%"
popd
endlocal & exit /b %exit_code%
