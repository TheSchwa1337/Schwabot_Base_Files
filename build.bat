@echo off
echo Building Schwabot Brain Trading Executable...
echo.

echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo Building executable...
pyinstaller schwabot.spec --clean --noconfirm

echo.
if exist "dist\SchwabotBrainTrader.exe" (
    echo [SUCCESS] Build successful! Executable created at: dist\SchwabotBrainTrader.exe
    echo.
    echo Testing executable...
    cd dist
    SchwabotBrainTrader.exe
) else (
    echo [ERROR] Build failed!
)

pause
