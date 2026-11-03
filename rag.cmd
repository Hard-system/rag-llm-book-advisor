@echo off
rem Convenience wrapper: run `rag ...` from this folder (or anywhere, once this folder is on PATH).
set PYTHONUTF8=1
"%~dp0.venv\Scripts\rag.exe" %*
