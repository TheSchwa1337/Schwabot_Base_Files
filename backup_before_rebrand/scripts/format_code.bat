@echo off
echo 🚀 Auto-formatting Schwabot Code with PEP 8 Standards
echo ====================================================

echo.
echo 🎨 Running Black formatter...
black core/ utils/ --line-length=100 --target-version=py39
if %errorlevel% neq 0 (
    echo ❌ Black formatting failed
) else (
    echo ✅ Black formatting completed
)

echo.
echo 📦 Running isort for import sorting...
isort core/ utils/ --profile=black --line-length=100 --atomic
if %errorlevel% neq 0 (
    echo ❌ Import sorting failed
) else (
    echo ✅ Import sorting completed
)

echo.
echo 🔍 Running flake8 linting check...
flake8 core/ utils/ --max-line-length=100 --extend-ignore=E203,W503 --count --statistics
if %errorlevel% neq 0 (
    echo ⚠️ Some linting issues found
) else (
    echo ✅ All files pass linting
)

echo.
echo 🎉 Code formatting completed!
pause 