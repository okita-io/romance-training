#Requires -Version 5.1
<#
.SYNOPSIS
  Install Python dependencies for train_qwen_unsloth.py (Unsloth + PyTorch CUDA).

.DESCRIPTION
  Requires Python 3.10 or newer (Unsloth depends on peft>=0.18, which needs 3.10+).
  Python 3.12 is the most reliable for ML wheels; 3.13 often has no packages on the legacy cu121 index.

  Mirrors Dockerfile.training:
  - PyTorch (default cu126, per pytorch.org stable; use cu118 if you only have CUDA 11.8 on PATH)
  - Unsloth from GitHub (pulls trl/peft per unsloth-zoo — do not install trl<0.9)
  - accelerate, bitsandbytes; optional xformers
  - datasets, transformers, huggingface-hub, tokenizers; NumPy>=2.1.3

.PARAMETER Python
  Python executable (default: python).

.PARAMETER SkipXformers
  Skip xformers (useful on Windows if the pinned wheel fails to install).

.PARAMETER CreateVenv
  Create a virtual environment at -VenvPath (if missing) using -Python as the base interpreter, then install into it.

.PARAMETER VenvPath
  Directory for the venv, relative to this script (default: .venv).

.PARAMETER TorchCuda
  PyTorch index: cu126/cu128 (current stable builds), cu118 (CUDA 11.8 toolkit on PATH), cu121 (legacy; often empty for Python 3.13+ on Windows), or cpu.

.EXAMPLE
  .\install_training_deps.ps1
.EXAMPLE
  .\install_training_deps.ps1 -TorchCuda cu118
.EXAMPLE
  .\install_training_deps.ps1 -TorchCuda cu128
.EXAMPLE
  .\install_training_deps.ps1 -CreateVenv
.EXAMPLE
  .\install_training_deps.ps1 -CreateVenv -Python py -3.12
.EXAMPLE
  .\install_training_deps.ps1 -Python .\.venv\Scripts\python.exe
.EXAMPLE
  .\install_training_deps.ps1 -SkipXformers
#>
param(
    [string] $Python = "python",
    [switch] $SkipXformers,
    [switch] $CreateVenv,
    [string] $VenvPath = ".venv",
    [ValidateSet("cu128", "cu126", "cu121", "cu118", "cpu")]
    [string] $TorchCuda = "cu126"
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

if ($CreateVenv) {
    $BootstrapPython = $Python
    $VenvFull = Join-Path $RepoRoot $VenvPath
    $VenvPython = Join-Path $VenvFull "Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating venv at $VenvFull (using $BootstrapPython)..."
        & $BootstrapPython -m venv $VenvFull
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed" }
    }
    else {
        Write-Host "Using existing venv at $VenvFull"
    }
    $Python = $VenvPython
}

# Unsloth -> peft>=0.18 requires Python 3.10+
& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Unsloth needs Python 3.10 or newer (current packages require peft>=0.18)."
    Write-Host "Interpreter: $Python"
    & $Python --version
    Write-Host ""
    Write-Host "Fix:"
    Write-Host "  1. Install Python 3.10+ from https://www.python.org/downloads/"
    Write-Host "  2. If .venv was created with an old Python, remove it: Remove-Item -Recurse -Force .venv"
    Write-Host "  3. Re-run with a 3.10+ binary, e.g.:"
    Write-Host "       .\install_training_deps.ps1 -CreateVenv -Python py -3.12"
    throw "Python 3.10+ required"
}

& $Python -c "import sys; raise SystemExit(0 if sys.version_info[:2] < (3, 13) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Python 3.13+: prefer cu126/cu128 (not cu121 — pip may report 'No matching distribution'). Use https://pytorch.org/get-started/locally/ to confirm the wheel line for your OS."
}
if ($TorchCuda -eq "cu121") {
    & $Python -c "import sys; raise SystemExit(0 if sys.version_info < (3, 13) else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "cu121 + Python 3.13+ often has NO torch wheels on Windows. Aborting cu121; re-run with -TorchCuda cu126 (or cu118 / Python 3.12)."
        throw "Use -TorchCuda cu126 or cu118 instead of cu121 for this Python version"
    }
}

function Invoke-Pip {
    param([Parameter(Mandatory)][string[]] $PipArgs)
    & $Python -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed: $Python -m pip $($PipArgs -join ' ')"
    }
}

Write-Host "Using interpreter: $Python"
& $Python --version

Write-Host "`n[1/6] Upgrading pip, setuptools, wheel..."
Invoke-Pip @("install", "--upgrade", "pip", "setuptools", "wheel")

$torchIndex = switch ($TorchCuda) {
    "cu128" { "https://download.pytorch.org/whl/cu128" }
    "cu126" { "https://download.pytorch.org/whl/cu126" }
    "cu121" { "https://download.pytorch.org/whl/cu121" }
    "cu118" { "https://download.pytorch.org/whl/cu118" }
    "cpu"   { "https://download.pytorch.org/whl/cpu" }
}
Write-Host "`n[2/6] Installing PyTorch (index: $TorchCuda -> $torchIndex)..."
Invoke-Pip @(
    "install", "torch", "torchvision", "torchaudio",
    "--index-url", $torchIndex
)

Write-Host "`n[3/6] Installing Unsloth from Git..."
Invoke-Pip @("install", "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")

Write-Host "`n[4/6] Installing accelerate & bitsandbytes (trl/peft stay on Unsloth's versions)..."
Invoke-Pip @("install", "accelerate", "bitsandbytes")
Write-Host "  Aligning trl with unsloth-zoo (>=0.18.2, <=0.24.0, !=0.19.0)..."
Invoke-Pip @("install", "trl>=0.18.2,<=0.24.0,!=0.19.0")
if (-not $SkipXformers) {
    Write-Host "  Trying optional xformers (may fail on some Windows setups)..."
    & $Python -m pip install xformers
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "xformers not installed; training may still work without it."
    }
}
else {
    Write-Host "  (Skipping xformers: -SkipXformers)"
}

Write-Host "`n[5/6] Installing Hugging Face datasets stack..."
Invoke-Pip @("install", "datasets", "transformers", "huggingface-hub", "tokenizers")

Write-Host "`n[6/6] Ensuring NumPy and Pillow wheels match this Python (fixes common PIL/_imaging errors)..."
Invoke-Pip @("install", "--upgrade", "--force-reinstall", "numpy>=2.1.3", "pillow")

if ($CreateVenv) {
    $ActivateRel = Join-Path $VenvPath "Scripts\Activate.ps1"
    Write-Host "`nDone. Activate:  .\$ActivateRel"
    Write-Host "Then run:  python train_qwen_unsloth.py"
}
else {
    Write-Host "`nDone. Activate your venv if you use one, then run: $Python train_qwen_unsloth.py"
}
if ($env:OS -match "Windows") {
    Write-Host ""
    Write-Host "If import torch fails with WinError 126 (torch_python.dll):"
    Write-Host "  - Install MSVC++ x64 redistributable (latest)."
    Write-Host "  - CUDA/driver must match the wheel (cu126/cu128 need recent drivers; CUDA 11.8 toolkit only -> -TorchCuda cu118)."
    Write-Host "  - Run: .\diagnose_torch_windows.ps1 -Python $Python"
    Write-Host "  - conda deactivate if (base) and venv are both active."
}
