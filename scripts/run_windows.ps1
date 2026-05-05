param(
    [string]$Python = "python",
    [string]$DatasetsRoot = "..\Datasets",
    [string]$SourceCheckpoint = "checkpoints\source.pt",
    [int]$SourceEpochs = 60,
    [int]$TargetEpochs = 50,
    [int]$BatchSize = 128
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "==> Train source domain: AWGN.dat"
& $Python scripts\train_source.py `
    --data-root (Join-Path $DatasetsRoot "AWGN.dat") `
    --output $SourceCheckpoint `
    --epochs $SourceEpochs `
    --batch-size $BatchSize

$targets = @(
    "Rayleigh1.dat",
    "Rayleigh3.dat",
    "Rician1.dat",
    "Rician3.dat"
)

foreach ($target in $targets) {
    Write-Host "==> Adapt target domain: $target"
    & $Python scripts\adapt_target.py `
        --data-root (Join-Path $DatasetsRoot $target) `
        --source-checkpoint $SourceCheckpoint `
        --target-split all `
        --eval-split all `
        --epochs $TargetEpochs `
        --batch-size $BatchSize
}

Write-Host "==> Done. Results are saved in results\"
