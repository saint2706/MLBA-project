@echo off
REM ================================================================
REM 🎬 Game of Thrones AI Script Generator - Windows Launch Script
REM ================================================================
REM Optimized Windows batch file for easy setup and execution
REM Last updated: August 2025

setlocal enabledelayedexpansion

REM ================================================================
REM 🎨 Setup Color Variables (if supported)
REM ================================================================
set "GREEN=[32m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "PURPLE=[35m"
set "CYAN=[36m"
set "RED=[31m"
set "NC=[0m"

REM ================================================================
REM 📋 Default Configuration
REM ================================================================
set "VENV_NAME=venv"
set "PYTHON_BIN=python"
set "REQ_FILE=requirements.txt"
set "MAIN_SCRIPT=main_modern.py"
set "ENHANCED_SCRIPT=modern_example_usage.py"
set "DATA_PATH=data\Game_of_Thrones_Script.csv"
set "EPOCHS=200"
set "BATCH_SIZE=16"
set "CONTEXT_WINDOW=64"
set "MODE=train"
set "USE_ENHANCED=false"
set "FORCE_CPU=false"
set "QUICK=false"

REM ================================================================
REM 🔧 Parse Command Line Arguments
REM ================================================================
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--mode" (
    set "MODE=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--data" (
    set "DATA_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--epochs" (
    set "EPOCHS=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--enhanced" (
    set "USE_ENHANCED=true"
    set "MAIN_SCRIPT=%ENHANCED_SCRIPT%"
    shift
    goto :parse_args
)
if "%~1"=="--cpu" (
    set "FORCE_CPU=true"
    shift
    goto :parse_args
)
if "%~1"=="--quick" (
    set "EPOCHS=50"
    set "QUICK=true"
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
echo Unknown option: %~1
goto :show_help

:args_done

REM ================================================================
REM 🆘 Help Function
REM ================================================================
:show_help
echo.
echo %CYAN%🎬 Game of Thrones AI Script Generator%NC%
echo %CYAN%=====================================%NC%
echo.
echo %YELLOW%Usage:%NC% run_project.bat [OPTIONS]
echo.
echo %YELLOW%Options:%NC%
echo   --mode MODE         Action to perform (train^|generate^|dashboard^|analyze)
echo   --data PATH         Path to Game of Thrones dataset CSV
echo   --epochs NUM        Number of training epochs (default: 200)
echo   --enhanced          Use enhanced training (modern_example_usage.py)
echo   --cpu               Force CPU-only training
echo   --quick             Quick training (50 epochs)
echo   --help, -h          Show this help message
echo.
echo %YELLOW%Modes:%NC%
echo   train       🚀 Train the AI model (default)
echo   generate    🎭 Generate dialogue using existing model
echo   dashboard   📊 Create training visualization dashboard
echo   analyze     📈 Analyze the dataset statistics
echo.
echo %YELLOW%Examples:%NC%
echo   run_project.bat                    # Standard training
echo   run_project.bat --enhanced         # Best quality training
echo   run_project.bat --mode generate    # Generate dialogue
echo   run_project.bat --quick            # Quick 50-epoch training
echo   run_project.bat --mode dashboard   # Create visualizations
echo.
if "%~1"=="--help" exit /b 0
if "%~1"=="-h" exit /b 0
exit /b 1

REM ================================================================
REM 🖥️ System Information Display
REM ================================================================
echo %PURPLE%🎬 Game of Thrones AI Script Generator%NC%
echo %PURPLE%=====================================%NC%
echo %BLUE%📊 Configuration:%NC%
echo    Mode: %YELLOW%!MODE!%NC%
echo    Dataset: %YELLOW%!DATA_PATH!%NC%
echo    Epochs: %YELLOW%!EPOCHS!%NC%
echo    Script: %YELLOW%!MAIN_SCRIPT!%NC%
echo.

REM ================================================================
REM 🐍 Python and Virtual Environment Setup
REM ================================================================
echo %BLUE%🐍 Setting up Python environment...%NC%

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Python not found. Please install Python 3.10 or 3.11%NC%
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo    Python version: %YELLOW%!PYTHON_VERSION!%NC%

REM Create virtual environment if it doesn't exist
if not exist "%VENV_NAME%" (
    echo %YELLOW%📦 Creating virtual environment: %VENV_NAME%%NC%
    python -m venv "%VENV_NAME%"
    echo %GREEN%✅ Virtual environment created%NC%
)

REM Activate virtual environment
echo %YELLOW%🔌 Activating virtual environment...%NC%
call "%VENV_NAME%\Scripts\activate.bat"
echo %GREEN%✅ Virtual environment activated%NC%

REM Upgrade pip
echo %YELLOW%⬆️  Upgrading pip...%NC%
python -m pip install --upgrade pip setuptools wheel --quiet

REM ================================================================
REM 🎯 GPU Detection
REM ================================================================
echo.
echo %BLUE%🚀 Detecting GPU and CUDA...%NC%

set "USE_CUDA=false"
if "%FORCE_CPU%"=="false" (
    nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
    if not errorlevel 1 (
        set "USE_CUDA=true"
        echo %GREEN%✅ NVIDIA GPU detected%NC%
        for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do echo    GPU: %YELLOW%%%i%NC%
    ) else (
        echo %YELLOW%⚠️  No NVIDIA GPU detected - using CPU%NC%
    )
) else (
    echo %YELLOW%🖥️  CPU-only mode forced%NC%
)

REM ================================================================
REM 📦 Install Dependencies
REM ================================================================
echo.
echo %BLUE%📦 Installing dependencies...%NC%

if exist "requirements.txt" (
    echo    Using: %YELLOW%requirements.txt%NC%
    
    REM Install PyTorch with appropriate CUDA support
    if "%USE_CUDA%"=="true" (
        echo %YELLOW%🔥 Installing PyTorch with CUDA support...%NC%
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    ) else (
        echo %YELLOW%🖥️  Installing CPU-only PyTorch...%NC%
        pip install torch torchvision torchaudio --quiet
    )
    
    echo %YELLOW%📚 Installing remaining packages...%NC%
    pip install -r requirements.txt --quiet
    
) else (
    echo %YELLOW%⚠️  No requirements file found, installing core packages...%NC%
    
    if "%USE_CUDA%"=="true" (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    ) else (
        pip install torch torchvision torchaudio --quiet
    )
    pip install transformers datasets pandas numpy plotly kaleido --quiet
)

echo %GREEN%✅ Dependencies installed%NC%

REM ================================================================
REM 🧪 Test Installation
REM ================================================================
echo.
echo %BLUE%🧪 Testing installation...%NC%

python -c "import sys, torch, transformers, pandas as pd, numpy as np, plotly; print(f'✅ Python: {sys.version.split()[0]}'); print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}'); print(f'✅ Transformers: {transformers.__version__}'); print(f'✅ All packages working!')"

REM ================================================================
REM 🎯 Execute Main Action
REM ================================================================
echo.
echo %BLUE%🎯 Executing main action: %YELLOW%!MODE!%NC%

if "!MODE!"=="train" (
    echo %GREEN%🚀 Starting AI training...%NC%
    echo    This may take several hours. Training progress will be saved to training_output.txt
    echo    You can monitor progress with: %YELLOW%Get-Content training_output.txt -Wait -Tail 10%NC%
    
    if not exist "!MAIN_SCRIPT!" (
        echo %RED%❌ Script not found: !MAIN_SCRIPT!%NC%
        pause
        exit /b 1
    )
    
    python "!MAIN_SCRIPT!"
    
) else if "!MODE!"=="generate" (
    echo %GREEN%🎭 Generating dialogue...%NC%
    python -c "from modern_example_usage import quick_generate; print('=== JON SNOW ==='); print(quick_generate('jon snow: ', character='jon snow', max_length=150)); print('\n=== TYRION LANNISTER ==='); print(quick_generate('tyrion: ', character='tyrion', max_length=150)); print('\n=== DAENERYS TARGARYEN ==='); print(quick_generate('daenerys: ', character='daenerys', max_length=150))"
    
) else if "!MODE!"=="dashboard" (
    echo %GREEN%📊 Creating training dashboard...%NC%
    python -c "from modern_plot import quick_dashboard; quick_dashboard()"
    echo %GREEN%✅ Dashboard created! Check the generated .html files%NC%
    
) else if "!MODE!"=="analyze" (
    echo %GREEN%📈 Analyzing dataset...%NC%
    if not exist "!DATA_PATH!" (
        echo %RED%❌ Dataset not found: !DATA_PATH!%NC%
        pause
        exit /b 1
    )
    python -c "from improved_helperAI import analyze_dataset; import json; result = analyze_dataset('!DATA_PATH!'); print(json.dumps(result, indent=2))"
    
) else (
    echo %RED%❌ Unknown mode: !MODE!%NC%
    goto :show_help
)

REM ================================================================
REM 🎉 Completion Message
REM ================================================================
echo.
echo %GREEN%🎉 Operation completed successfully!%NC%
echo %BLUE%📁 Generated files:%NC%
dir /b *.pt *.pkl *.html *.txt 2>nul || echo    No generated files found yet

echo.
echo %YELLOW%💡 Next steps:%NC%
if "!MODE!"=="train" (
    echo    1. Check training_output.txt for detailed logs
    echo    2. Generate visualizations: %CYAN%run_project.bat --mode dashboard%NC%
    echo    3. Test generation: %CYAN%run_project.bat --mode generate%NC%
) else if "!MODE!"=="generate" (
    echo    1. Try different characters and creativity settings
    echo    2. Generate longer dialogue by modifying max_length
) else if "!MODE!"=="dashboard" (
    echo    1. Open the generated .html files in your web browser
    echo    2. Explore the interactive visualizations
) else if "!MODE!"=="analyze" (
    echo    1. Review the dataset statistics above
    echo    2. Start training: %CYAN%run_project.bat --mode train%NC%
)

echo.
echo %PURPLE%🐉⚔️👑 Winter is coming, but your AI is ready! 🐺❄️%NC%
pause
