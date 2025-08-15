#!/usr/bin/env bash
# run_project.sh - Cross-platform setup and launcher for Modern TV Script Generation
set -e

# Defaults
VENV_NAME="venv"
PYTHON_BIN="python"
REQ_FILE="requirements.txt"
MAIN_SCRIPT="main_modern.py"
DATA_PATH="data/Game_of_Thrones_Script.csv"
CONTEXT_WINDOW=1024
GENERATE_ONLY=false

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case $1 in
    --data) DATA_PATH="$2"; shift 2 ;;
    --context) CONTEXT_WINDOW="$2"; shift 2 ;;
    --generate) GENERATE_ONLY=true; shift 1 ;;
    --help|-h)
      echo "Usage: ./run_project.sh [--data path] [--context num_tokens] [--generate]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=== Modern TV Script Generation ==="
echo "Dataset : $DATA_PATH"
echo "Context window : $CONTEXT_WINDOW"
echo "Generate Only : $GENERATE_ONLY"
echo "==================================="

# -------------------------------
# OS detection for venv activation
# -------------------------------
OS_TYPE="$(uname | tr '[:upper:]' '[:lower:]')"
if [[ "$OS_TYPE" == "linux" || "$OS_TYPE" == "darwin" ]]; then
  VENV_ACTIVATE="$VENV_NAME/bin/activate"
else
  VENV_ACTIVATE="$VENV_NAME/Scripts/activate"
fi

# -------------------------------
# Create and activate venv
# -------------------------------
if [ ! -d "$VENV_NAME" ]; then
  echo "Creating venv: $VENV_NAME"
  $PYTHON_BIN -m venv "$VENV_NAME"
fi
source "$VENV_ACTIVATE"

# -------------------------------
# Upgrade pip
# -------------------------------
echo "Upgrading pip..."
pip install --upgrade pip

# -------------------------------
# Detect optimal CUDA/PyTorch Wheel using nvcc
# -------------------------------
USE_CUDA121=false
USE_CUDA118=false

if command -v nvcc >/dev/null 2>&1; then
  echo "Checking CUDA version using nvcc..."
  NVCC_OUTPUT=$(nvcc --version | grep "release" | sed 's/.*release //;s/,.*//')
  echo "Detected CUDA Toolkit version: $NVCC_OUTPUT"

  CUDA_MAJOR=$(echo "$NVCC_OUTPUT" | cut -d. -f1)
  CUDA_MINOR=$(echo "$NVCC_OUTPUT" | cut -d. -f2)

  # Map CUDA versions to PyTorch wheels
  if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 1 ]]; then
    USE_CUDA121=true
  elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
    USE_CUDA118=true
  fi
else
  echo "nvcc not found; defaulting to CPU PyTorch..."
fi

# -------------------------------
# Install dependencies
# -------------------------------
if [ -f "$REQ_FILE" ]; then
  echo "requirements.txt found; installing from file."

  # Remove torch packages if already present
  pip uninstall -y torch torchvision torchaudio || true

  if [ "$USE_CUDA121" = true ]; then
    echo "Installing PyTorch (CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  elif [ "$USE_CUDA118" = true ]; then
    echo "Installing PyTorch (CUDA 11.8)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  else
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
  fi

  # Install the rest from requirements.txt excluding torch lines
  grep -Ev '^torch|^torchvision|^torchaudio' "$REQ_FILE" | xargs -n 1 pip install
else
  echo "No requirements.txt found. Installing core packages..."
  if [ "$USE_CUDA121" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  elif [ "$USE_CUDA118" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  else
    pip install torch torchvision torchaudio
  fi
  pip install transformers datasets sentencepiece pandas numpy plotly kaleido pytest
fi

# -------------------------------
# Check CUDA availability in PyTorch
# -------------------------------
echo "Checking PyTorch CUDA support..."
python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.get_device_name(0))
EOF

# -------------------------------
# Run main script
# -------------------------------
if [ ! -f "$MAIN_SCRIPT" ]; then
  echo "ERROR: $MAIN_SCRIPT not found."
  exit 1
fi

if [ "$GENERATE_ONLY" = true ]; then
  echo "Skipping training; running generation only..."
  $PYTHON_BIN "$MAIN_SCRIPT" --data "$DATA_PATH" --context "$CONTEXT_WINDOW" --generate
else
  $PYTHON_BIN "$MAIN_SCRIPT" --data "$DATA_PATH" --context "$CONTEXT_WINDOW"
fi
