$log = Join-Path $PSScriptRoot "step500_monitor.log"
$target = 501
while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    try {
        $stepLine = ssh -o ConnectTimeout=15 okita-pc "grep -oE '[0-9]+/3000' /home/okita/git_repos/romance-training/train/training.log | tail -1"
        $status = ssh -o ConnectTimeout=15 okita-pc "docker ps --filter name=mistral_style_train --format '{{.Status}}'"
        $ckpts = ssh -o ConnectTimeout=15 okita-pc "ls /home/okita/git_repos/romance-training/train/mistral_style_lora/ 2>/dev/null"
        $step = 0
        if ($stepLine -match '^(\d+)/3000$') { $step = [int]$Matches[1] }
        $line = "$ts step=$step status=$status"
        Add-Content -Path $log -Value $line
        if ($step -ge $target) {
            Add-Content -Path $log -Value "CLEARED_STEP_500 at $ts checkpoints=$ckpts"
            break
        }
        if ($status -notmatch 'Up') {
            Add-Content -Path $log -Value "CONTAINER_DOWN at $ts status=$status"
            break
        }
    } catch {
        Add-Content -Path $log -Value "$ts ERROR $($_.Exception.Message)"
    }
    Start-Sleep -Seconds 300
}
