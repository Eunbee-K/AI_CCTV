# 웹 데모(serve_web.py)를 Cloudflare Quick Tunnel로 외부에 공개한다.
# Quick Tunnel은 재실행할 때마다 주소가 바뀌므로, 현재 주소를 public_url.txt에 기록해둔다.
# 사용법: 이 폴더에서 우클릭 > PowerShell로 실행, 또는 `powershell -File start_public_demo.ps1`

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$urlFile = Join-Path $root "public_url.txt"
$logFile = Join-Path $root "tunnel.log"

if (-not (Test-Path $cloudflared)) {
    Write-Output "[demo] cloudflared를 찾을 수 없습니다: $cloudflared"
    exit 1
}

$listening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Output "[demo] 8000번 포트에 웹 서버가 없어 새 창에서 serve_web.py를 실행합니다..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; python serve_web.py --port 8000"
    Start-Sleep -Seconds 3
} else {
    Write-Output "[demo] 8000번 포트에 이미 웹 서버가 떠 있습니다. 그대로 사용합니다."
}

"(연결 중...)" | Out-File $urlFile -Encoding utf8 -NoNewline
Write-Output "[demo] Cloudflare Tunnel 시작..."
Start-Process -FilePath $cloudflared -ArgumentList "tunnel", "--url", "http://localhost:8000", "--no-autoupdate" -RedirectStandardError $logFile

$deadline = (Get-Date).AddSeconds(30)
$url = $null
while ((Get-Date) -lt $deadline -and -not $url) {
    Start-Sleep -Milliseconds 500
    if (Test-Path $logFile) {
        $match = Select-String -Path $logFile -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) { $url = $match.Matches[0].Value }
    }
}

if ($url) {
    $url | Out-File $urlFile -Encoding utf8 -NoNewline
    Write-Output "[demo] 공개 주소: $url"
    Write-Output "[demo] 이 주소는 매번 $urlFile 에 갱신되어 저장됩니다."
} else {
    "(연결 실패 - tunnel.log 확인)" | Out-File $urlFile -Encoding utf8 -NoNewline
    Write-Output "[demo] 주소를 못 찾았습니다. $logFile 을 확인하세요."
}
