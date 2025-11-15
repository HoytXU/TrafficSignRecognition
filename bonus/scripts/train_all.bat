@echo off
REM Batch Training Script for Windows
REM Trains all models with specified hyperparameters

REM Define hyperparameters
set EPOCH=10
set LR=0.001
set WEIGHT_DECAY=0.001
set BATCH_SIZE=128

REM Define model architectures
set MODELS=lenet resnet18 vgg16 alexnet squeezenet1_0 vit_b_16 my_net

REM Get script directory
set SCRIPT_DIR=%~dp0
set BONUS_DIR=%SCRIPT_DIR%..
set PROJECT_ROOT=%BONUS_DIR%..

REM Create logs directory
if not exist "%BONUS_DIR%\logs" mkdir "%BONUS_DIR%\logs"

REM Train each model
for %%i in (%MODELS%) do (
    set LOG_FILE=%BONUS_DIR%\logs\%%i-%EPOCH%-%LR%-%BATCH_SIZE%-%WEIGHT_DECAY%.log
    echo ==========================================
    echo Training %%i...
    echo Log file: %LOG_FILE%
    echo ==========================================
    
    cd /d "%PROJECT_ROOT%"
    python "%BONUS_DIR%\training\train.py" --epoch %EPOCH% --lr %LR% --batch_size %BATCH_SIZE% --weight_decay %WEIGHT_DECAY% --model %%i > "%LOG_FILE%" 2>&1
    
    if %ERRORLEVEL% EQU 0 (
        echo ✓ %%i training completed successfully
    ) else (
        echo ✗ %%i training failed. Check log: %LOG_FILE%
    )
    echo.
)

echo All training jobs completed!
pause

