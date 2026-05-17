$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$OutRoot = "outputs\moe_val_no_leak_ETTh1_ep100"
$LogDir = Join-Path $OutRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stdout = Join-Path $LogDir "run.out.log"
$Stderr = Join-Path $LogDir "run.err.log"

$Args = @(
    "run", "-n", "my_fram", "python", "-u", "scripts\run_moe_val_search.py",
    "--base-config", "configs\ETTh1.yaml",
    "--epochs", "100",
    "--out-root", $OutRoot,
    "--device", "cuda:0",
    "--variants",
    "full_penalty_gate",
    "level_amp_gate_amp_std_loose",
    "level_amp_gate_amp_std_frac85",
    "aug_level_amp_gate_amp_std",
    "aug_level_gate_amp",
    "aug_level_range_signed_noclip",
    "level_amp_scale"
)

Start-Process `
    -FilePath "F:\Anaconda3\Scripts\conda.exe" `
    -ArgumentList $Args `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru
