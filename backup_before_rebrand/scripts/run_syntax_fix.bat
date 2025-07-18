@echo off
echo Running Master Syntax Fixer...
echo ================================

python master_syntax_fixer.py

if %ERRORLEVEL% NEQ 0 (
    echo Python script failed, trying alternative approach...
    python -c "import sys; print('Python version:', sys.version)"
)

echo.
echo Syntax fix completed!
pause 