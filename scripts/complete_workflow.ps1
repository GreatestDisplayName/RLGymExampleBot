<#
.SYNOPSIS
    Manages the complete RLGym training and execution workflow.
.DESCRIPTION
    This PowerShell script automates the entire process for an RLGym bot:
    - Activates the Python virtual environment.
    - Installs dependencies from requirements.txt.
    - Runs the main workflow script (src/complete_workflow.py) with specified parameters.
    - Handles training, testing, model conversion, and gameplay simulation.
.PARAMETER Quick
    A switch to perform a quick test run with a small number of timesteps (10,000).
.PARAMETER Agent
    Specifies the reinforcement learning algorithm to use. Valid options are "PPO", "SAC", "TD3". Defaults to "PPO".
.PARAMETER Timesteps
    The total number of timesteps for training. Defaults to 100,000.
.PARAMETER TestEpisodes
    The number of episodes to run for testing the trained model. Defaults to 5.
.PARAMETER Render
    A switch to enable rendering during the testing phase.
.PARAMETER NoConvert
    A switch to skip the model conversion step after training.
.PARAMETER NoPlay
    A switch to skip the final play-testing phase.
.EXAMPLE
    . \complete_workflow.ps1 -Quick -Render
    Runs a quick training workflow with rendering enabled for the test phase.
.EXAMPLE
    . \complete_workflow.ps1 -Agent SAC -Timesteps 500000
    Runs the workflow using the SAC algorithm for 500,000 timesteps.
.NOTES
    Author: Gemini
    Last Modified: 2025-08-10
#>
[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [switch]$Quick,
    [ValidateSet("PPO", "SAC", "TD3")]
    [string]$Agent = "PPO",
    [int]$Timesteps = 100000,
    [int]$TestEpisodes = 5,
    [switch]$Render,
    [switch]$NoConvert,
    [switch]$NoPlay
)

# --- Configuration ---
$VenvPath = Join-Path $PSScriptRoot ".\.venv"
$RequirementsFile = Join-Path $PSScriptRoot ".\requirements.txt"
$ScriptPath = Join-Path $PSScriptRoot ".\src\complete_workflow.py"

# --- Functions ---
function Invoke-Step {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,
        [Parameter(Mandatory=$true)]
        [scriptblock]$ScriptBlock
    )
    Write-Host ("-`-" * 20)
    Write-Host "🔵 STEP: $Name"
    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $ScriptBlock
        $Stopwatch.Stop()
        Write-Host ("✅ SUCCESS: $Name completed in {0:N2} seconds." -f $Stopwatch.Elapsed.TotalSeconds) -ForegroundColor Green
    } catch {
        $Stopwatch.Stop()
        Write-Host ("❌ ERROR: $Name failed after {0:N2} seconds." -f $Stopwatch.Elapsed.TotalSeconds) -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

# --- Main Script ---
Clear-Host
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🚀 RLGym Complete Workflow Manager 🚀" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

Invoke-Step -Name "Virtual Environment Activation" -ScriptBlock {
    if (Test-Path -Path "$VenvPath\Scripts\Activate.ps1") {
        . "$VenvPath\Scripts\Activate.ps1"
        Write-Host "Virtual environment activated."
    } else {
        throw "Virtual environment not found at '$VenvPath'. Please run setup first."
    }
    $pythonVersion = python --version 2>&1
    Write-Host "Using Python: $($pythonVersion -join ' `n')"
}

Invoke-Step -Name "Dependency Installation" -ScriptBlock {
    if (Test-Path $RequirementsFile) {
        Write-Host "Installing dependencies from $RequirementsFile..."
        pip install -r $RequirementsFile --upgrade
    } else {
        Write-Host "Warning: '$RequirementsFile' not found. Skipping dependency installation." -ForegroundColor Yellow
    }
}

Invoke-Step -Name "Workflow Execution" -ScriptBlock {
    $pythonArgs = @(
        "--agent", $Agent,
        "--timesteps", $Timesteps,
        "--test-episodes", $TestEpisodes
    )
    if ($Quick) { $pythonArgs += "--quick" }
    if ($Render) { $pythonArgs += "--render" }
    if ($NoConvert) { $pythonArgs += "--no-convert" }
    if ($NoPlay) { $pythonArgs += "--no-play" }

    Write-Host "Starting Python workflow script: $ScriptPath"
    Write-Host "Arguments: $($pythonArgs -join ' ')"
    if ($PSCmdlet.ShouldProcess("Executing the main Python workflow script.", "python $ScriptPath $($pythonArgs -join ' ')")) {
        $process = Start-Process -FilePath "python" -ArgumentList "-u", "$ScriptPath", $pythonArgs -Wait -PassThru -NoNewWindow
        if ($process.ExitCode -ne 0) {
            throw "Python script exited with error code $($process.ExitCode)."
        }
    }
}

Write-Host ""
Write-Host "🎉🎉🎉 All workflow steps completed successfully! 🎉🎉🎉" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"