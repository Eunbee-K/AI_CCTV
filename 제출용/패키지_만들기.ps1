# 제출 패키지를 다시 만든다.
#
# app/ 은 apps/AI_CCTV 의 사본이라 저장소가 바뀌면 낡는다. 제출 직전에 한 번
# 돌려서 최신 코드·모델로 맞춘 뒤 압축하면 된다.
#
# 사용:  powershell -ExecutionPolicy Bypass -File 패키지_만들기.ps1
#        powershell -ExecutionPolicy Bypass -File 패키지_만들기.ps1 -Zip

param([switch]$Zip)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path (Split-Path -Parent $here) "apps\AI_CCTV"
$dst  = Join-Path $here "app"

if (-not (Test-Path (Join-Path $src "serve_demo.py"))) {
    Write-Error "원본을 찾을 수 없습니다: $src"
}

Write-Host "원본: $src"
Write-Host "대상: $dst"
Write-Host ""

if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $dst "assets") | Out-Null

Copy-Item (Join-Path $src "backend") -Destination (Join-Path $dst "backend") -Recurse -Force
Copy-Item (Join-Path $src "ui")      -Destination (Join-Path $dst "ui")      -Recurse -Force
Copy-Item (Join-Path $src "serve_demo.py")     -Destination $dst -Force
Copy-Item (Join-Path $src "requirements.txt")  -Destination $dst -Force
# 처음 실행 때 필요한 패키지를 깔아주는 준비 스크립트. 이것이 빠지면
# 받는 PC에서 ModuleNotFoundError 로 끝난다(예전 제출본의 실패 원인).
Copy-Item (Join-Path $src "setup_env.py")      -Destination $dst -Force

# 배포에 필요한 모델만 넣는다. best_test5/test6 은 옛 버전이라 뺀다(98MB 절약).
foreach ($m in @("best.pt", "classifier.onnx", "classifier.classes.json")) {
    Copy-Item (Join-Path $src "assets\$m") -Destination (Join-Path $dst "assets") -Force
}

# 파이썬 캐시는 넣지 않는다 — 다른 PC에서 쓸모없고 용량만 는다.
Get-ChildItem $dst -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# .venv 는 만들어진 PC의 절대경로가 박혀 있어 다른 PC에서 못 쓴다.
# 받는 쪽에서 setup_env.py 가 새로 만든다. 담지 않는다.
$venv = Join-Path $dst ".venv"
if (Test-Path $venv) { Remove-Item $venv -Recurse -Force }

$mb = [math]::Round((Get-ChildItem $here -Recurse -File |
      Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "패키지 준비 완료 — $mb MB"

if ($Zip) {
    # .NET 은 PowerShell 의 현재 위치를 모른다. 상대경로가 섞이면
    # "Illegal characters in path" 로 죽으므로 절대경로로 확정한다.
    $outDir  = (Resolve-Path (Split-Path -Parent $here)).ProviderPath
    $zipPath = [System.IO.Path]::Combine($outDir, "하수관로CCTV자동분석_제출.zip")
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }

    # Compress-Archive 는 한글 경로에서 "Illegal characters in path" 로 죽는다.
    # .NET ZipFile 을 직접 쓴다. 이 스크립트 자신은 빼고 담는다 — 받는 사람에게
    # 필요한 것은 실행하기.bat / README.md / app 뿐이다.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $staging = [System.IO.Path]::Combine(
        $env:TEMP, "submit_" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Get-ChildItem $here -Exclude "패키지_만들기.ps1" |
        Copy-Item -Destination $staging -Recurse -Force
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $staging, $zipPath,
        [System.IO.Compression.CompressionLevel]::Optimal, $false)
    Remove-Item -LiteralPath $staging -Recurse -Force

    $zmb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-Host "압축 완료: $zipPath ($zmb MB)"
} else {
    Write-Host ""
    Write-Host "압축까지 하려면: powershell -ExecutionPolicy Bypass -File 패키지_만들기.ps1 -Zip"
}
