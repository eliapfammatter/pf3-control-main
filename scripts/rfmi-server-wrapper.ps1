# SimsenRFMI Server Auto-Restart Wrapper
# This script automatically restarts the RFMI server when disconnections occur
#
# Features:
# - Automatically starts and initializes the server
# - Monitors for "Disconnection" in server output
# - Kills all child processes spawned by this server instance (Sim23, etc.)
# - Uses process tree tracking (parent-child relationships) to only kill
#   processes belonging to THIS server, not other running programs
# - Robust error handling for edge cases
# - Automatic restart after disconnection

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
    Read-Host "Press Enter to exit"
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
    Read-Host "Press Enter to exit"
    exit 1
}

$RestartCount = 0

while ($true) {
    $RestartCount++

    Write-Host ""
    Write-Host "[$RestartCount] Starting SimsenRFMI Server..." -ForegroundColor Green
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""

    # Start the process with redirected I/O
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ServerPath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.WorkingDirectory = Split-Path $ServerPath
    $psi.CreateNoWindow = $false

    $process = [System.Diagnostics.Process]::Start($psi)

    # Wait for process to initialize
    Start-Sleep -Milliseconds 800

    # Send Enter key to start the server
    Write-Host "Sending Enter key to start server..." -ForegroundColor Cyan
    try {
        $process.StandardInput.WriteLine()
        $process.StandardInput.Flush()
    }
    catch {
        Write-Host "Warning: Could not send Enter key: $_" -ForegroundColor Yellow
    }

    Write-Host "Server started (PID: $($process.Id))" -ForegroundColor Green
    Write-Host "Monitoring for disconnection..." -ForegroundColor Cyan
    Write-Host ("-" * 50)

    # Monitor output for "Disconnection"
    $disconnectionDetected = $false

    try {
        # Read output in a loop
        while (-not $process.HasExited) {
            # Check stdout
            while (-not $process.StandardOutput.EndOfStream) {
                $line = $process.StandardOutput.ReadLine()
                if ($line) {
                    Write-Host $line

                    # Check for disconnection
                    if ($line -match "(?i)disconnection") {
                        $disconnectionDetected = $true
                        Write-Host ""
                        Write-Host ("=" * 50) -ForegroundColor Yellow
                        Write-Host ">>> DISCONNECTION DETECTED! <<<" -ForegroundColor Yellow -BackgroundColor Red
                        Write-Host ("=" * 50) -ForegroundColor Yellow
                        Write-Host ""
                        break
                    }
                }
            }

            if ($disconnectionDetected) {
                break
            }

            # Small delay to prevent CPU spinning
            Start-Sleep -Milliseconds 100
        }

        # If process exited without disconnection message
        if (-not $disconnectionDetected -and $process.HasExited) {
            Write-Host ""
            Write-Host ">>> SERVER PROCESS EXITED <<<" -ForegroundColor Yellow
            Write-Host ""
        }

    }
    catch {
        Write-Host "Error monitoring output: $_" -ForegroundColor Red
    }

    # Kill all child processes spawned by this server
    Write-Host "Checking for hanging child processes..." -ForegroundColor Yellow

    try {
        # Get all descendant processes of the server
        $serverPid = $process.Id

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
        $descendants = @(Get-ChildProcesses -ParentId $serverPid)

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

    # Kill the server process if still running
    if (-not $process.HasExited) {
        Write-Host "Stopping server process (PID: $($process.Id))..." -ForegroundColor Yellow
        try {
            $process.Kill()
            $process.WaitForExit(5000)
            Write-Host "Server process stopped." -ForegroundColor Green
        }
        catch {
            Write-Host "Server process already stopped." -ForegroundColor Gray
        }
    }

    $process.Dispose()

    Write-Host ""
    Write-Host "Restarting in 2 seconds..." -ForegroundColor Yellow
    Write-Host ""
    Start-Sleep -Seconds 2
}
