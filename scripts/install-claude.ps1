<#
.SYNOPSIS
  Connects book-rag to Claude Code:
   1. registers the MCP server "book-rag" for your user (available in every project/folder)
   2. installs the /use-rag skill into ~/.claude/skills so it works everywhere
   3. (optional) adds this folder to your PATH so `rag` works in any terminal
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install-claude.ps1 -AddToPath
#>
param([switch]$AddToPath)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run scripts\setup.ps1 first ($Py not found)." }

if (-not (Get-Command claude -ErrorAction SilentlyContinue)) { throw "The 'claude' CLI was not found on PATH." }

Write-Host "== Registering MCP server 'book-rag' (user scope)" -ForegroundColor Cyan
try { claude mcp remove book-rag --scope user 2>$null | Out-Null } catch {}
claude mcp add book-rag --scope user -e "RAG_HOME=$Root" -e "PYTHONUTF8=1" -- "$Py" -m book_rag.mcp_server
if (-not $?) { throw "claude mcp add failed" }

Write-Host "== Installing /use-rag skill into ~/.claude/skills" -ForegroundColor Cyan
$SkillDst = Join-Path $env:USERPROFILE ".claude\skills\use-rag"
New-Item -ItemType Directory -Force $SkillDst | Out-Null
Copy-Item (Join-Path $Root ".claude\skills\use-rag\SKILL.md") $SkillDst -Force

if ($AddToPath) {
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (($userPath -split ";") -notcontains $Root) {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$Root", "User")
        Write-Host "Added $Root to your user PATH (open a new terminal to use 'rag' anywhere)."
    }
}

Write-Host "`nDone. Restart Claude Code, then check with:  claude mcp list" -ForegroundColor Green
Write-Host "In any Claude Code chat type:  /use-rag how do I build better habits?"
