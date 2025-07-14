@echo off
echo ========================================
echo Schwabot Trading System - Windows Installer
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8+ from https://python.org
    echo After installing Python, run this installer again.
    pause
    exit /b 1
)

echo Python found. Checking version...
python --version

:: Create virtual environment
echo.
echo Creating virtual environment...
python -m venv schwabot-env
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call schwabot-env\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing Schwabot dependencies...
pip install -r ..\..\requirements.txt

:: Install Schwabot
echo Installing Schwabot...
pip install -e ..\..

:: Create desktop shortcut
echo Creating desktop shortcut...
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Schwabot.lnk'); $Shortcut.TargetPath = '%~dp0schwabot-env\Scripts\python.exe'; $Shortcut.Arguments = '-m schwabot'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Save()"

:: Create start menu entry
echo Creating start menu entry...
if not exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Schwabot" mkdir "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Schwabot"
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%APPDATA%\Microsoft\Windows\Start Menu\Programs\Schwabot\Schwabot.lnk'); $Shortcut.TargetPath = '%~dp0schwabot-env\Scripts\python.exe'; $Shortcut.Arguments = '-m schwabot'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Save()"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Schwabot has been installed successfully.
echo.
echo To start Schwabot:
echo 1. Double-click the desktop shortcut, OR
echo 2. Run: schwabot-env\Scripts\activate && python -m schwabot
echo.
echo For more information, see the documentation in docs/
echo.
pause 