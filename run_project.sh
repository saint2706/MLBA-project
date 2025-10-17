#!/usr/bin/env bash
# ================================================================
# 🎬 Game of Thrones AI Script Generator - Launch Script
# ================================================================
# Optimized cross-platform setup and launcher
# Supports Windows (WSL/Git Bash), Linux, and macOS
# Last updated: August 2025

set -e  # Exit on any error

# ================================================================
# 🎨 Colors for better terminal output
# ================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ================================================================
# 📋 Default Configuration
# ================================================================
VENV_NAME="venv"
PYTHON_BIN="python"
REQ_FILE="requirements.txt"
MAIN_MODULE="got_script_generator.main_modern"
ENHANCED_MODULE="got_script_generator.modern_example_usage"
DATA_PATH="data/Game_of_Thrones_Script.csv"
EPOCHS=200
BATCH_SIZE=16
CONTEXT_WINDOW=64
MODE="train"  # Options: train, generate, dashboard, analyze

# ================================================================
# 🆘 Help Function
# ================================================================
show_help() {
    echo -e "${CYAN}🎬 Game of Thrones AI Script Generator${NC}"
    echo -e "${CYAN}=====================================${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC} ./run_project.sh [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --mode MODE         Action to perform (train|generate|dashboard|analyze)"
    echo "  --data PATH         Path to Game of Thrones dataset CSV"
    echo "  --epochs NUM        Number of training epochs (default: 200)"
    echo "  --batch-size NUM    Training batch size (default: 16)"
    echo "  --context NUM       Context window size (default: 64)"
    echo "  --enhanced          Use enhanced training (got_script_generator.modern_example_usage)"
    echo "  --cpu               Force CPU-only training"
    echo "  --quick             Quick training (50 epochs)"
    echo "  --help, -h          Show this help message"
    echo ""
    echo -e "${YELLOW}Modes:${NC}"
    echo "  train       🚀 Train the AI model (default)"
    echo "  generate    🎭 Generate dialogue using existing model"
    echo "  dashboard   📊 Create training visualization dashboard"
    echo "  analyze     📈 Analyze the dataset statistics"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./run_project.sh                    # Standard training"
    echo "  ./run_project.sh --enhanced         # Best quality training"
    echo "  ./run_project.sh --mode generate    # Generate dialogue"
    echo "  ./run_project.sh --quick            # Quick 50-epoch training"
    echo "  ./run_project.sh --mode dashboard   # Create visualizations"
}

# ================================================================
# 🔧 Parse Command Line Arguments
# ================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --data) DATA_PATH="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --context) CONTEXT_WINDOW="$2"; shift 2 ;;
        --enhanced) MAIN_MODULE="$ENHANCED_MODULE"; shift 1 ;;
        --cpu) FORCE_CPU=true; shift 1 ;;
        --quick) EPOCHS=50; shift 1 ;;
        --help|-h) show_help; exit 0 ;;
        *) echo -e "${RED}❌ Unknown option: $1${NC}"; show_help; exit 1 ;;
    esac
done

# ================================================================
# 🖥️ System Detection and Setup
# ================================================================
echo -e "${PURPLE}🎬 Game of Thrones AI Script Generator${NC}"
echo -e "${PURPLE}=====================================${NC}"
echo -e "${BLUE}📊 Configuration:${NC}"
echo -e "   Mode: ${YELLOW}$MODE${NC}"
echo -e "   Dataset: ${YELLOW}$DATA_PATH${NC}"
echo -e "   Epochs: ${YELLOW}$EPOCHS${NC}"
echo -e "   Batch size: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "   Context window: ${YELLOW}$CONTEXT_WINDOW${NC}"
echo -e "   Module: ${YELLOW}$MAIN_MODULE${NC}"
echo ""

# Detect OS for proper venv activation
OS_TYPE="$(uname 2>/dev/null | tr '[:upper:]' '[:lower:]' || echo 'windows')"
if [[ "$OS_TYPE" == "linux" || "$OS_TYPE" == "darwin" ]]; then
    VENV_ACTIVATE="$VENV_NAME/bin/activate"
    PLATFORM="Unix"
else
    VENV_ACTIVATE="$VENV_NAME/Scripts/activate"
    PLATFORM="Windows"
fi

echo -e "${BLUE}🖥️  Platform: ${YELLOW}$PLATFORM${NC}"

# ================================================================
# 🐍 Python and Virtual Environment Setup
# ================================================================
echo -e "\n${BLUE}🐍 Setting up Python environment...${NC}"

# Check Python version
if ! command -v python >/dev/null 2>&1; then
    echo -e "${RED}❌ Python not found. Please install Python 3.10 or 3.11${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo -e "   Python version: ${YELLOW}$PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment: $VENV_NAME${NC}"
    python -m venv "$VENV_NAME"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}🔌 Activating virtual environment...${NC}"
source "$VENV_ACTIVATE"
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Ensure project package is importable
PYTHONPATH_ADD="$(pwd)/src"
export PYTHONPATH="${PYTHONPATH_ADD}${PYTHONPATH:+:$PYTHONPATH}"

# Upgrade pip
echo -e "${YELLOW}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel --quiet

# ================================================================
# 🎯 CUDA Detection and PyTorch Installation
# ================================================================
echo -e "\n${BLUE}🚀 Detecting GPU and CUDA...${NC}"

CUDA_VERSION=""
USE_CUDA=false

if [ "$FORCE_CPU" != "true" ]; then
    # Check for nvidia-smi first
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
        
        # Try to get CUDA version from nvidia-smi
        if nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1; then
            USE_CUDA=true
            echo -e "   GPU: ${YELLOW}$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)${NC}"
        fi
        
        # Check nvcc for CUDA toolkit version
        if command -v nvcc >/dev/null 2>&1; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //;s/,.*//')
            echo -e "   CUDA Toolkit: ${YELLOW}$CUDA_VERSION${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No NVIDIA GPU detected - using CPU${NC}"
    fi
else
    echo -e "${YELLOW}🖥️  CPU-only mode forced${NC}"
fi

# ================================================================
# 📦 Install Dependencies
# ================================================================
echo -e "\n${BLUE}📦 Installing dependencies...${NC}"

# Use optimized requirements if available, otherwise fall back to regular
if [ -f "requirements_optimized.txt" ]; then
    REQ_FILE="requirements_optimized.txt"
    echo -e "   Using: ${YELLOW}$REQ_FILE${NC}"
elif [ -f "$REQ_FILE" ]; then
    echo -e "   Using: ${YELLOW}$REQ_FILE${NC}"
else
    echo -e "${YELLOW}⚠️  No requirements file found, installing core packages...${NC}"
fi

# Install PyTorch with appropriate CUDA support
if [ "$USE_CUDA" = "true" ] && [ "$FORCE_CPU" != "true" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    if [[ "$CUDA_MAJOR" -ge 12 ]]; then
        echo -e "${YELLOW}🔥 Installing PyTorch with CUDA 12.1 support...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        echo -e "${YELLOW}🔥 Installing PyTorch with CUDA 11.8 support...${NC}"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    fi
else
    echo -e "${YELLOW}🖥️  Installing CPU-only PyTorch...${NC}"
    pip install torch torchvision torchaudio --quiet
fi

# Install remaining packages
if [ -f "$REQ_FILE" ]; then
    echo -e "${YELLOW}📚 Installing remaining packages...${NC}"
    # Install non-torch packages from requirements
    grep -Ev '^torch|^torchvision|^torchaudio|^#|^$' "$REQ_FILE" | while read package; do
        if [[ -n "$package" && "$package" != *"--"* ]]; then
            pip install "$package" --quiet
        fi
    done
else
    # Install essential packages manually
    pip install transformers datasets pandas numpy plotly kaleido --quiet
fi

echo -e "${GREEN}✅ Dependencies installed${NC}"

# ================================================================
# 🧪 Test Installation
# ================================================================
echo -e "\n${BLUE}🧪 Testing installation...${NC}"

python - <<'EOF'
import sys
import torch
import transformers
import pandas as pd
import numpy as np
import plotly

print(f"✅ Python: {sys.version.split()[0]}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ Pandas: {pd.__version__}")
print(f"✅ NumPy: {np.__version__}")
print(f"✅ Plotly: {plotly.__version__}")
EOF

# ================================================================
# 🎯 Execute Main Action
# ================================================================
echo -e "\n${BLUE}🎯 Executing main action: ${YELLOW}$MODE${NC}"

case $MODE in
    "train")
        echo -e "${GREEN}🚀 Starting AI training...${NC}"
        echo -e "   This may take several hours. Training progress will be saved to training_output.txt"
        echo -e "   You can monitor progress with: ${YELLOW}tail -f training_output.txt${NC}"
        
        if ! python - <<PY; then
import importlib
import sys

try:
    importlib.import_module("${MAIN_MODULE}")
except ModuleNotFoundError:
    sys.exit(1)
PY
        then
            echo -e "${RED}❌ Module not found: ${MAIN_MODULE}${NC}"
            exit 1
        fi

        python -m "${MAIN_MODULE}"
        ;;

    "generate")
        echo -e "${GREEN}🎭 Generating dialogue...${NC}"
        python -c "
from got_script_generator.modern_example_usage import quick_generate
print('=== JON SNOW ===')
print(quick_generate('jon snow: ', character='jon snow', max_length=150))
print('\n=== TYRION LANNISTER ===')
print(quick_generate('tyrion: ', character='tyrion', max_length=150))
print('\n=== DAENERYS TARGARYEN ===')
print(quick_generate('daenerys: ', character='daenerys', max_length=150))
"
        ;;

    "dashboard")
        echo -e "${GREEN}📊 Creating training dashboard...${NC}"
        python -c "from got_script_generator.modern_plot import quick_dashboard; quick_dashboard()"
        echo -e "${GREEN}✅ Dashboard created! Check the generated .html files${NC}"
        ;;

    "analyze")
        echo -e "${GREEN}📈 Analyzing dataset...${NC}"
        if [ ! -f "$DATA_PATH" ]; then
            echo -e "${RED}❌ Dataset not found: $DATA_PATH${NC}"
            exit 1
        fi
        python -c "
from got_script_generator.improved_helperAI import analyze_dataset
import json
result = analyze_dataset('$DATA_PATH')
print(json.dumps(result, indent=2))
"
        ;;
        
    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        show_help
        exit 1
        ;;
esac

# ================================================================
# 🎉 Completion Message
# ================================================================
echo -e "\n${GREEN}🎉 Operation completed successfully!${NC}"
echo -e "${BLUE}📁 Generated files:${NC}"
ls -la *.pt *.pkl *.html *.txt 2>/dev/null || echo "   No generated files found yet"

echo -e "\n${YELLOW}💡 Next steps:${NC}"
case $MODE in
    "train")
        echo -e "   1. Check training_output.txt for detailed logs"
        echo -e "   2. Generate visualizations: ${CYAN}./run_project.sh --mode dashboard${NC}"
        echo -e "   3. Test generation: ${CYAN}./run_project.sh --mode generate${NC}"
        ;;
    "generate")
        echo -e "   1. Try different characters and creativity settings"
        echo -e "   2. Generate longer dialogue by modifying max_length"
        ;;
    "dashboard")
        echo -e "   1. Open the generated .html files in your web browser"
        echo -e "   2. Explore the interactive visualizations"
        ;;
    "analyze")
        echo -e "   1. Review the dataset statistics above"
        echo -e "   2. Start training: ${CYAN}./run_project.sh --mode train${NC}"
        ;;
esac

echo -e "\n${PURPLE}🐉⚔️👑 Winter is coming, but your AI is ready! 🐺❄️${NC}"
