@echo off
REM Batch Training Script for Windows
REM Wrapper script that calls the Python training script

REM Get script directory
set SCRIPT_DIR=%~dp0
set BONUS_DIR=%SCRIPT_DIR%..
set PROJECT_ROOT=%BONUS_DIR%..

cd /d "%PROJECT_ROOT%"

REM Use Python script for better cross-platform support
python "%BONUS_DIR%\scripts\train_all_models.py" %*
pause

