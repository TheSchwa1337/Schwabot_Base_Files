@echo off
setlocal enabledelayedexpansion

echo Starting Code Quality Checks...

REM Install required tools
echo [*] Installing required tools
pip install autoflake black isort flake8 chardet

REM Prepare files
echo [*] Preparing files for code quality check
python prepare_flake8.py

REM Run comprehensive code check
echo [*] Running comprehensive code check
python comprehensive_code_check.py

REM Run flake8 check
echo [*] Running flake8 check
flake8 core/ schwabot/ utils/ config/ --max-line-length=100 --count

REM Check code formatting with black
echo [*] Checking code formatting with black
black core/ schwabot/ utils/ config/ --check --line-length=100

REM Check import sorting
echo [*] Checking import sorting
isort core/ schwabot/ utils/ config/ --check-only

REM Generate final report
echo [*] Generating code quality report
python final_code_quality_check.py

REM Check the exit status
if %errorlevel% equ 0 (
    echo Code Quality Checks Passed Successfully!
    exit /b 0
) else (
    echo Code Quality Checks Failed. Please review the report.
    exit /b 1
) 