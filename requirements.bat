@echo off
REM requirements.bat - Windows helper to create virtualenv and install dependencies
REM Usage: Open an elevated (or normal) command prompt in the repository root and run:
REM   requirements.bat

:: Create virtual environment directory .venv if it doesn't exist
python -m venv .venv
if %ERRORLEVEL% NEQ 0 (
  echo Failed to create virtual environment. Ensure Python is on PATH.
  pause
  exit /b 1
)

:: Activate the venv for the current cmd session
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
  echo Failed to activate the virtual environment.
  pause
  exit /b 1
)

:: Upgrade pip and install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
  echo pip install failed. Check the output above.
  pause
  exit /b 1
)

echo Dependencies installed successfully in .venv
pause
