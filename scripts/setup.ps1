<#
.SYNOPSIS
  One-time setup: creates .venv, installs book-rag, downloads the models and runs a self-test.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#>
param(
    [string]$Python = "3.13",
    [switch]$SkipModels
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "== book-rag setup in $Root" -ForegroundColor Cyan

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    Write-Host "Using uv to create .venv (Python $Python)"
    uv venv .venv --python $Python
    if (-not $?) { throw "uv venv failed" }
    # --native-tls uses the Windows certificate store (works behind corporate proxies / antivirus TLS inspection)
    uv pip install --native-tls --python .venv\Scripts\python.exe -e ".[dev]" truststore
    if (-not $?) { throw "dependency install failed" }
} else {
    Write-Host "uv not found - using python -m venv + pip"
    py -$Python -m venv .venv
    if (-not $?) { python -m venv .venv }
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -e ".[dev]" truststore
    if (-not $?) { throw "dependency install failed" }
}

New-Item -ItemType Directory -Force library, data | Out-Null

if (-not $SkipModels) {
    Write-Host "`n== Downloading models and running self-test (first run downloads ~2-4 GB, once)" -ForegroundColor Cyan
    .venv\Scripts\rag.exe doctor
}

Write-Host "`nDone. Next steps:" -ForegroundColor Green
Write-Host "  1. Put books into $Root\library   (sub-folders = categories)"
Write-Host "  2. .\rag sync                        (index them)"
Write-Host "  3. .\rag search `"your question`"     (test from the terminal)"
Write-Host "  4. powershell -ExecutionPolicy Bypass -File scripts\install-claude.ps1   (connect to Claude Code)"
