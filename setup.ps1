# OmniVoice 本地 TTS 服务 · 自举脚本
# 在全新 Windows 机器上从 git 克隆后运行：重建内置 Python 运行时、安装依赖、拉取模型权重。
# 用法：powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PyDir = Join-Path $Root "python"
$SitePackages = Join-Path $PyDir "Lib\site-packages"
$EmbedVer = "3.13.2"
$EmbedUrl = "https://www.python.org/ftp/python/$EmbedVer/python-$EmbedVer-embed-amd64.zip"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# 1) 下载并解压 embeddable Python
if (Test-Path (Join-Path $PyDir "python.exe")) {
    Write-Step "内置 Python 已存在，跳过下载 ($PyDir)"
} else {
    Write-Step "下载 embeddable Python $EmbedVer"
    $zip = Join-Path $env:TEMP "py-embed.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Invoke-WebRequest -Uri $EmbedUrl -OutFile $zip
    Write-Step "解压到 $PyDir"
    New-Item -ItemType Directory -Force -Path $PyDir | Out-Null
    Expand-Archive -Path $zip -DestinationPath $PyDir -Force
    Remove-Item $zip -Force
}

# 2) 配置 python313._pth 启用 site-packages
Write-Step "配置 python313._pth"
$pth = Join-Path $PyDir "python313._pth"
$pthContent = @(
    "python313.zip",
    ".",
    "Lib/site-packages",
    "import site"
) -join "`n"
Set-Content -Path $pth -Value $pthContent -Encoding ASCII
New-Item -ItemType Directory -Force -Path $SitePackages | Out-Null

$Py = Join-Path $PyDir "python.exe"

# 3) 安装 pip（embed 版不带 pip）+ 依赖
Write-Step "确保 pip 可用"
& $Py -m ensurepip --upgrade 2>$null
if (-not $?) {
    # 退化方案：用 get-pip.py
    $gp = Join-Path $env:TEMP "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $gp
    & $Py $gp
}

Write-Step "安装 Python 依赖（torch/torchaudio 走 PyTorch CUDA 12.8 源）"
$Req = Join-Path $Root "requirements.txt"
& $Py -m pip install -r $Req --extra-index-url https://download.pytorch.org/whl/cu128

# 4) 拉取模型权重
Write-Step "拉取 OmniVoice 模型权重"
& $Py (Join-Path $Root "download_model.py")

Write-Step "完成。运行 start.cmd 启动服务，或在 DSHT 按 README 配置配音接口。"
