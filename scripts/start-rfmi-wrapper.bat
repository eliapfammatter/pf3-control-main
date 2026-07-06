@echo off
REM Launch the RFMI Server Auto-Restart Wrapper using PowerShell

echo Starting SimsenRFMI Server Auto-Restart Wrapper (PowerShell)...
echo.

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0rfmi-server-wrapper.ps1"

pause
