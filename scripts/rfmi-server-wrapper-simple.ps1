# SimsenRFMIServer Auto-Restart Wrapper (Simple Version)
# Monitors process and restarts when it exits

# Get SIMSEN installation path from environment variable
$SimsenFmiPackage = $env:SIMSEN_FMI_PACKAGE

if (-not $SimsenFmiPackage) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "SimsenRFMI Server Auto-Restart Wrapper" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "ERROR: SIMSEN_FMI_PACKAGE environment variable not set!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set the SIMSEN_FMI_PACKAGE environment variable to your SIMSEN installation directory." -ForegroundColor Yellow
    Write-Host "Example: C:\Users\lives\SIMSEN\SIMSEN_4_0_3_2025_05" -ForegroundColor Gray
    Write-Host ""
    pause
    exit 1
}

$ServerPath = Join-Path $SimsenFmiPackage "exe\SimsenRFMIServer.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SimsenRFMI Server Auto-Restart Wrapper" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "SIMSEN_FMI_PACKAGE: $SimsenFmiPackage" -ForegroundColor Gray
Write-Host "Server path: $ServerPath" -ForegroundColor Gray
Write-Host ""

# Verify server executable exists
if (-not (Test-Path $ServerPath)) {
    Write-Host "ERROR: SimsenRFMIServer.exe not found at:" -ForegroundColor Red
    Write-Host $ServerPath -ForegroundColor Red
    Write-Host ""
    Write-Host "Please verify your SIMSEN_FMI_PACKAGE environment variable is correct." -ForegroundColor Yellow
    pause
    exit 1
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinAPI {
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);

    public const int VK_RETURN = 0x0D;
    public const int KEYEVENTF_KEYUP = 0x0002;
}
"@

$RestartCount = 0

while ($true) {
    $RestartCount++

    Write-Host ""
    Write-Host "[$RestartCount] Starting SimsenRFMI Server..." -ForegroundColor Green
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""

    # Start the server in a new window
    $process = Start-Process -FilePath $ServerPath `
                             -WorkingDirectory (Split-Path $ServerPath) `
                             -PassThru

    # Wait for process to initialize
    Start-Sleep -Milliseconds 1000

    # Try to send Enter key to the process
    Write-Host "Sending Enter key to start server..." -ForegroundColor Cyan

    try {
        # Get the main window handle
        if ($process.MainWindowHandle -ne 0) {
            [WinAPI]::SetForegroundWindow($process.MainWindowHandle)
            Start-Sleep -Milliseconds 200

            # Simulate Enter key press
            [WinAPI]::keybd_event([WinAPI]::VK_RETURN, 0, 0, 0)
            Start-Sleep -Milliseconds 50
            [WinAPI]::keybd_event([WinAPI]::VK_RETURN, 0, [WinAPI]::KEYEVENTF_KEYUP, 0)
        } else {
            Write-Host "Warning: Could not find main window. Please press Enter manually." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "Warning: Could not send Enter key automatically. Please press Enter manually." -ForegroundColor Yellow
    }

    Write-Host "Server is running (PID: $($process.Id))" -ForegroundColor Green
    Write-Host "Monitoring process... Will restart when it exits." -ForegroundColor Cyan
    Write-Host ""

    # Monitor the process
    $process.WaitForExit()

    Write-Host ""
    Write-Host ">>> SERVER EXITED (Disconnection detected) <<<" -ForegroundColor Yellow -BackgroundColor Red
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""

    # Kill all child processes spawned by this server
    Write-Host "Checking for hanging child processes..." -ForegroundColor Yellow

    try {
        # Function to get all child processes recursively
        function Get-ChildProcesses {
            param([int]$ParentId)

            $children = @()
            try {
                $directChildren = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $ParentId }

                foreach ($child in $directChildren) {
                    $children += $child
                    # Recursively get grandchildren
                    $children += Get-ChildProcesses -ParentId $child.ProcessId
                }
            }
            catch {
                # Silently continue if process tree changed
            }

            return $children
        }

        # Get all descendants of the server process
        $descendants = @(Get-ChildProcesses -ParentId $process.Id)

        if ($descendants) {
            Write-Host "Found $($descendants.Count) child process(es) to clean up:" -ForegroundColor Yellow

            foreach ($child in $descendants) {
                Write-Host "  - Killing $($child.Name) (PID: $($child.ProcessId))..." -ForegroundColor Yellow

                try {
                    Stop-Process -Id $child.ProcessId -Force -ErrorAction Stop
                    Write-Host "    Successfully killed $($child.Name)" -ForegroundColor Green
                }
                catch {
                    Write-Host "    Could not kill PID $($child.ProcessId) (may have already exited)" -ForegroundColor Gray
                }
            }
        }
        else {
            Write-Host "No child processes found." -ForegroundColor Green
        }
    }
    catch {
        Write-Host "Warning: Could not check for child processes: $_" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Restarting in 2 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
