#Requires -Version 5.1
<#
.SYNOPSIS
  Install Windows build prerequisites for Unsloth GGUF export (llama.cpp).

.DESCRIPTION
  Unsloth builds llama.cpp under %USERPROFILE%\.unsloth\llama.cpp using CMake and
  the Visual Studio 2022 C++ toolchain. Automatic install often fails if winget
  cannot elevate or if CMake is missing from PATH in the same shell.

  Run this from an elevated PowerShell if winget install fails (Right-click - Run as administrator).

  After it finishes: close the terminal, open a NEW one, verify cmake --version,
  then rerun: python train_qwen_unsloth.py --config train_config.toml --export-only

.NOTES
  Optional: clone llama.cpp into this repo as .\llama.cpp - Unsloth will prefer a
  working copy there when you run Python from the repo root (see unsloth_zoo.llama_cpp).

  Override install location: set env UNSLOTH_LLAMA_CPP_PATH to your llama.cpp folder.
#>
$ErrorActionPreference = "Continue"

function Add-CMakeToUserPath {
    $pf86 = ${env:ProgramFiles(x86)}
    $candidates = @(
        "$env:ProgramFiles\CMake\bin",
        "$pf86\CMake\bin"
    )
    foreach ($d in $candidates) {
        if (-not (Test-Path "$d\cmake.exe")) { continue }
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ([string]::IsNullOrEmpty($userPath)) { $userPath = "" }
        $already = ($userPath -split ';' | Where-Object { $_ -eq $d })
        if ($already) {
            Write-Host "CMake bin already on user PATH: $d" -ForegroundColor Green
        }
        else {
            [Environment]::SetEnvironmentVariable("Path", "$d;$userPath", "User")
            Write-Host "Added to user PATH (open NEW terminal for global effect): $d" -ForegroundColor Green
        }
        $pathParts = $env:Path -split ';'
        if ($pathParts -notcontains $d) {
            $env:Path = "$d;$env:Path"
            Write-Host "Prepended to this session PATH: $d" -ForegroundColor Green
        }
        return
    }
    Write-Host "cmake.exe not found under Program Files - reinstall Kitware.CMake or add bin manually." -ForegroundColor Yellow
}

function Invoke-WingetInstall {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [string[]]$ExtraArgs = @()
    )
    $wingetArgs = @(
        "install", "-e", "--id", $Id,
        "--accept-package-agreements", "--accept-source-agreements"
    ) + $ExtraArgs
    Write-Host ""
    Write-Host ">>> winget $($wingetArgs -join ' ')" -ForegroundColor Cyan
    & winget @wingetArgs
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host "OK: $Id" -ForegroundColor Green
    }
    elseif ($code -eq -1978335189) {
        Write-Host ('Already installed (winget reports present): ' + $Id) -ForegroundColor Yellow
    }
    else {
        Write-Host "winget exit $code for $Id - try elevated PowerShell or install manually." -ForegroundColor Yellow
    }
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host "winget not found. Install App Installer from the Microsoft Store, or install CMake + VS Build Tools manually."
    exit 1
}

Write-Host 'Unsloth GGUF prerequisites: CMake, OpenSSL dev headers, VS 2022 Build Tools + C++ workload'
Write-Host 'This can take 10-20+ minutes for Visual Studio components.'

Invoke-WingetInstall -Id "Kitware.CMake"
Add-CMakeToUserPath
Invoke-WingetInstall -Id "ShiningLight.OpenSSL.Dev"
Invoke-WingetInstall -Id "Microsoft.VisualStudio.2022.BuildTools" -ExtraArgs @(
    "--override",
    "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
)

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host '  1. Close this window and open a NEW PowerShell so PATH picks up CMake.'
Write-Host "  2. Run: cmake --version"
Write-Host "  3. From the repo root: python train_qwen_unsloth.py --config train_config.toml --export-only"
Write-Host ""
Write-Host "If winget still fails, run this script as Administrator, or install Kitware.CMake and"
Write-Host 'VS Build Tools manually from https://visualstudio.microsoft.com/downloads/ - Desktop development with C++.'
Write-Host ""
Write-Host "If Visual Studio Build Tools upgrade exits with code 1: open Visual Studio Installer," -ForegroundColor Yellow
Write-Host "Modify the 2022 Build Tools install, and ensure 'Desktop development with C++' is selected." -ForegroundColor Yellow
