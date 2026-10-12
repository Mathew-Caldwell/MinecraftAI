@echo off
setlocal
cd /d "%~dp0\.."
where python >nul 2>nul
if %errorlevel%==0 (
  set PYTHON_BOOTSTRAP=python
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    set PYTHON_BOOTSTRAP=py -3
  ) else (
    echo Python was not found. Install Python 3.11 or 3.12 from python.org, then run this again.
    exit /b 1
  )
)
%PYTHON_BOOTSTRAP% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e .
echo.
echo Setup complete. Use scripts\record.bat, scripts\train_bc.bat, or scripts\run_brain_dry.bat.
