Write-Host "Ensure you have activated the venv and run 'modal login' first."
Write-Host "Starting Modal distillation (will run modal_run.py run_full_distill)"

Write-Host "Running: modal run modal_run.py run_full_distill"
modal run modal_run.py run_full_distill

if ($LASTEXITCODE -ne 0) {
    Write-Host "Modal distill returned non-zero exit code: $LASTEXITCODE"
} else {
    Write-Host "Modal distill finished. Check ./checkpoints for outputs."
}
