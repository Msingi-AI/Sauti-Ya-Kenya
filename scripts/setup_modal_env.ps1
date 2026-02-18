<#
Setup Modal environment and dependencies locally.
Run this in PowerShell on your machine. It will:
 - create a Python venv
 - install the Modal SDK and project requirements
 - prompt you to run `modal login` to authenticate
#>

param(
    [string]$venvPath = ".venv"
)

Write-Host "Creating Python virtual environment at $venvPath..."
python -m venv $venvPath

Write-Host "Activating venv..."
& "$venvPath\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing Modal SDK and project requirements..."
pip install --upgrade pip
pip install modal
if (Test-Path "requirements-sauti.txt") {
    pip install -r requirements-sauti.txt
} else {
    Write-Host "requirements-sauti.txt not found; please ensure it's in repo root"
}

Write-Host "Now run: modal login  (this opens browser for authentication)"
Write-Host "After login, create a secret named 'hf-token' in the Modal dashboard (recommended) or via CLI."
Write-Host "To run precompute later: .\scripts\run_modal_precompute.ps1 --max-items 200"
