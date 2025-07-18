@echo off
setlocal enabledelayedexpansion

REM Ensure we're in the right directory
cd /d "%~dp0"

REM Install required tools
echo Installing required tools...
pip install autoflake black isort flake8

REM Step 1: Prepare files
echo Preparing files for flake8 check...
python prepare_flake8.py

REM Step 2: Run comprehensive flake8 check
echo Running comprehensive flake8 check...
python comprehensive_flake8_check.py

REM Check the exit code
if %errorlevel% equ 0 (
    echo ✅ Flake8 check completed successfully!
) else (
    echo ❌ Flake8 issues found. Please review the report.
    exit /b %errorlevel%
) 