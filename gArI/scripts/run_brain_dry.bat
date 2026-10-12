@echo off
setlocal
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m mcpvp_ai.brain --config configs\default.json --dry-run
