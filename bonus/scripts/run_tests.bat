@echo off
REM Test Runner Script for Bonus Level (Windows)

REM Get script directory
set SCRIPT_DIR=%~dp0
set BONUS_DIR=%SCRIPT_DIR%..
set PROJECT_ROOT=%BONUS_DIR%..

REM Change to project root
cd /d "%PROJECT_ROOT%"

echo ==========================================
echo Running Bonus Level Tests
echo ==========================================
echo.

REM Check if pytest is installed
where pytest >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing test dependencies...
    pip install -r "%BONUS_DIR%\requirements-test.txt"
)

REM Run tests
pytest "%BONUS_DIR%\tests\" -v --tb=short

REM Optionally run with coverage
if "%1"=="--coverage" (
    echo.
    echo ==========================================
    echo Running Tests with Coverage
    echo ==========================================
    pytest "%BONUS_DIR%\tests\" --cov="%BONUS_DIR%\core" --cov="%BONUS_DIR%\training" --cov="%BONUS_DIR%\evaluation" --cov-report=html --cov-report=term-missing
    echo.
    echo Coverage report generated in htmlcov/
)

echo.
echo ==========================================
echo Tests Complete
echo ==========================================
pause

