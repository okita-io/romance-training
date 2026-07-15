# Start vLLM with GGUF plugin for Phase 2 classification (3090 / Docker Desktop).
# Usage (repo root or train/vllm):
#   .\train\vllm\serve-classifier-3090.ps1
#   .\train\vllm\serve-classifier-3090.ps1 -Build

param(
    [switch]$Build,
    [string]$RepoId = "DavidAU/Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-GGUF",
    [string]$GgufFilename = "Llama-3.2-4X3B-MOE-Ultra-Instruct-10B-D_AU-Q4_k_m.gguf",
    [string]$Tokenizer = "unsloth/Llama-3.2-3B-Instruct",
    [string]$HfConfigPath = "unsloth/Llama-3.2-3B-Instruct",
    [string]$ServedName = "ultra-instruct-10b",
    [string]$Container = "vllm-classifier-3090",
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HfCache = Join-Path $env:USERPROFILE ".cache\huggingface"
$ModelDir = Join-Path $ScriptDir "models"
$LocalGguf = Join-Path $ModelDir $GgufFilename
New-Item -ItemType Directory -Force -Path $HfCache, $ModelDir | Out-Null

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not running. Start Docker Desktop, then retry."
}

if ($Build) {
    Write-Host "Building vllm-openai-gguf:latest ..."
    docker build -t vllm-openai-gguf:latest -f (Join-Path $ScriptDir "Dockerfile.gguf") $ScriptDir
}

if (-not (Test-Path $LocalGguf)) {
    Write-Host "Downloading $GgufFilename (~6 GB) ..."
    docker run --rm `
        -v "${HfCache}:/root/.cache/huggingface" `
        -v "${ModelDir}:/models" `
        --entrypoint python3 `
        vllm-openai-gguf:latest `
        -c "from huggingface_hub import hf_hub_download; import shutil; p=hf_hub_download(repo_id='$RepoId', filename='$GgufFilename'); shutil.copy2(p, '/models/$GgufFilename'); print(p)"
    if (-not (Test-Path $LocalGguf)) {
        throw "GGUF download failed: $LocalGguf not found"
    }
}

$existingContainer = docker ps -aq --filter "name=^/${Container}$"
if ($existingContainer) {
    docker rm -f $Container | Out-Null
}

Write-Host "Starting $Container on port $Port with local GGUF ..."
docker run -d `
    --name $Container `
    --gpus all `
    --ipc=host `
    --restart unless-stopped `
    -p "${Port}:8000" `
    -v "${HfCache}:/root/.cache/huggingface" `
    -v "${ModelDir}:/models:ro" `
    -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 `
    vllm-openai-gguf:latest `
    "/models/$GgufFilename" `
    --tokenizer $Tokenizer `
    --hf-config-path $HfConfigPath `
    --served-model-name $ServedName `
    --host 0.0.0.0 `
    --port 8000 `
    --max-num-seqs 4 `
    --max-model-len 8192 `
    --gpu-memory-utilization 0.90

Write-Host "Waiting for health at http://localhost:${Port}/health ..."
$deadline = (Get-Date).AddMinutes(30)
while ((Get-Date) -lt $deadline) {
    try {
        Invoke-RestMethod -Uri "http://localhost:${Port}/health" -TimeoutSec 5 | Out-Null
        break
    } catch {
        Start-Sleep -Seconds 10
    }
}

if ((Get-Date) -ge $deadline) {
    Write-Host "Timed out. Check logs: docker logs -f $Container"
    exit 1
}

Invoke-RestMethod -Uri "http://localhost:${Port}/v1/models" | ConvertTo-Json -Depth 6
Write-Host ""
Write-Host "Ready. Set LLM_BASE_URL=http://localhost:${Port}/v1 and LLM_MODEL=$ServedName"
