#Requires -Version 5.1
<#
.SYNOPSIS
  Diagnose WinError 126 when importing PyTorch on Windows (torch_python.dll / missing dependencies).

.PARAMETER Python
  Interpreter to inspect (default: python). Use .\.venv\Scripts\python.exe for this repo's venv.

.EXAMPLE
  .\diagnose_torch_windows.ps1 -Python .\.venv\Scripts\python.exe
#>
param(
    [string] $Python = "python"
)

$ErrorActionPreference = "Continue"

function Write-Section($title) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
}

Write-Section "Python"
try {
    & $Python --version
    $bits = & $Python -c "import struct; print(struct.calcsize('P') * 8)"
    Write-Host "Interpreter: $Python"
    Write-Host "Pointer width: $bits bit (need 64-bit for standard torch wheels)"
    $verOut = & $Python -c "import sys; print(sys.version)" 2>$null
    if ($verOut -match 'Anaconda|conda') {
        Write-Host "NOTE: This Python build mentions Anaconda/conda. Use a clean venv; avoid conda (base) active with pip installs, or prefer python.org installers to reduce DLL/PyTorch mix-ups." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Could not run $Python" -ForegroundColor Red
    exit 1
}

Write-Section "torch wheel layout"
try {
    $siteTorch = & $Python -c "import pathlib, sys; print(pathlib.Path(sys.prefix) / 'Lib' / 'site-packages' / 'torch' / 'lib')"
    if (-not (Test-Path $siteTorch)) {
        Write-Host "No torch installed under venv (missing folder): $siteTorch" -ForegroundColor Yellow
    }
    else {
        Write-Host "torch\lib: $siteTorch"
        $dll = Join-Path $siteTorch "torch_python.dll"
        Write-Host "torch_python.dll present: $(Test-Path $dll)"
    }
}
catch {
    Write-Host $_ -ForegroundColor Red
}

Write-Section "Visual C++ runtime (x64) - required for torch_python.dll"
$vc = @(
    "$env:SystemRoot\System32\vcruntime140.dll",
    "$env:SystemRoot\System32\vcruntime140_1.dll",
    "$env:SystemRoot\System32\msvcp140.dll",
    "$env:SystemRoot\System32\vcomp140.dll"
)
foreach ($f in $vc) {
    $ok = Test-Path $f
    $color = if ($ok) { "Green" } else { "Red" }
    Write-Host ("{0,-55} {1}" -f $f, $(if ($ok) { "OK" } else { "MISSING" })) -ForegroundColor $color
}
if (-not (Test-Path "$env:SystemRoot\System32\vcruntime140_1.dll")) {
    Write-Host "Install: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
}

Write-Section "CUDA 12.x runtime (for cu121 GPU wheels)"
$cudaRoots = @(
    "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA",
    "${env:ProgramFiles(x86)}\NVIDIA GPU Computing Toolkit\CUDA"
)
$foundCuda = $false
$foundCuda12 = $false
foreach ($root in $cudaRoots) {
    if (-not (Test-Path $root)) { continue }
    foreach ($verDir in @(Get-ChildItem $root -Directory -ErrorAction SilentlyContinue)) {
        $bin = Join-Path $verDir.FullName "bin"
        foreach ($cudart in @(Get-ChildItem $bin -Filter "cudart64*.dll" -ErrorAction SilentlyContinue)) {
            $foundCuda = $true
            Write-Host "FOUND $($cudart.FullName)"
            if ($cudart.Name -like 'cudart64_12*' -or $cudart.Name -like 'cudart64_13*') { $foundCuda12 = $true }
        }
    }
}
if (-not $foundCuda) {
    Write-Host "No cudart64*.dll found under Program Files CUDA folders." -ForegroundColor Yellow
    Write-Host "GPU wheels need a CUDA toolkit; add its bin folder to PATH."
    Write-Host 'Download: https://developer.nvidia.com/cuda-downloads'
}
elseif (-not $foundCuda12) {
    Write-Host ""
    Write-Host "MISMATCH: PyTorch cu126/cu128/cu121 GPU builds need CUDA 12.x runtime DLLs (e.g. cudart64_12*.dll)." -ForegroundColor Yellow
    Write-Host "You only have older CUDA (10.x/11.x). That often causes WinError 126 with torch_python.dll."
    Write-Host ""
    Write-Host "Pick one:"
    Write-Host "  A) Install CUDA Toolkit 12.x and put ...\CUDA\v12.x\bin on PATH ahead of older CUDA; use cu126/cu128 torch (pytorch.org)."
    Write-Host "  B) Match your GPU stack to CUDA 11.8 (typical):"
    Write-Host "       pip uninstall -y torch torchvision torchaudio"
    Write-Host "       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    Write-Host "     Or re-run: .\install_training_deps.ps1 -TorchCuda cu118 ..."
}

Write-Section "PATH entries mentioning CUDA or NVIDIA"
$paths = $env:PATH -split ";" | Where-Object { $_ -and ($_ -match 'CUDA|NVIDIA') }
if ($paths) { $paths | ForEach-Object { Write-Host $_ } }
else { Write-Host "(none)" -ForegroundColor Yellow }

Write-Section "Try import torch"
$torchCheckPy = Join-Path $env:TEMP "romance_factory_torch_check.py"
@'
import torch
v = torch.__version__
cu = torch.version.cuda
avail = torch.cuda.is_available()
print("torch.__version__", v)
print("torch.version.cuda", cu)
print("torch.cuda.is_available()", avail)
if "+cpu" in v:
    print("PROBLEM: CPU-only PyTorch (+cpu). CUDA toolkit install does not replace this; pip install GPU wheel from pytorch.org index.")
'@ | Set-Content -Path $torchCheckPy -Encoding UTF8
& $Python $torchCheckPy
Remove-Item -Path $torchCheckPy -Force -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) {
    Write-Host "Import failed (WinError 126: missing MSVC/CUDA DLL, or broken install)." -ForegroundColor Red
}
else {
    Write-Host "Import succeeded." -ForegroundColor Green
}

Write-Section "If you see +cpu or cuda_available False"
Write-Host "Reinstall GPU PyTorch (pick ONE index from https://pytorch.org/get-started/locally/):" -ForegroundColor Yellow
Write-Host '  pip uninstall -y torch torchvision torchaudio'
Write-Host '  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126'
Write-Host "  (CUDA 11.8 toolkit on PATH only: use .../whl/cu118)"
Write-Host "Then verify: python -c `"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())`""
Write-Host "You should see +cu126 (or similar) and cuda True, not +cpu."

Write-Section "Isolation test (CPU-only torch)"
Write-Host "If VC++ is OK but CUDA DLLs are missing, CPU wheels often still import."
Write-Host "Run manually:"
Write-Host "  pip uninstall -y torch torchvision torchaudio"
Write-Host "  pip install torch torchvision torchaudio"
Write-Host "  python -c `"import torch; print(torch.__version__)`""
Write-Host ""
Write-Host "If CPU import works, reinstall GPU build (cu126/cu128 per pytorch.org, or cu118 for CUDA 11.8 toolkit only):"
Write-Host "  pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
