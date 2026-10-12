@echo off
setlocal
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python -m mcpvp_ai.train_bc --config configs\default.json
