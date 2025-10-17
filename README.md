# 🎬 Game of Thrones AI Script Generator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Build an AI that speaks like the characters of *Game of Thrones*. The project packages
all training, generation, and visualization utilities under `src/got_script_generator`
while keeping helper scripts, datasets, and documentation accessible in the repository.

## 🚀 Quick Start

> The commands below assume a Unix-like shell. Replace the activation command with
> `venv\Scripts\activate` on Windows.

```bash
# 1. Clone and enter the repository
git clone https://github.com/saint2706/MLBA-project.git
cd MLBA-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. (Optional) Install full training dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the local package for smoke testing
pip install -e .

# 5. Run a lightweight smoke test
python -m got_script_generator.cli smoke-test --data data/Game_of_Thrones_Script.csv

# 6. Inspect dataset statistics (optional)
python -m got_script_generator.cli analyze-data --data data/Game_of_Thrones_Script.csv
```

These commands verify that the project package resolves correctly and that the bundled
Game of Thrones dialogue dataset is reachable. The smoke test does not launch a long
training run, making it safe to execute during setup.

## 🌿 Branch Workflow

Two sibling branches host the historical pipelines that inspired this consolidated
layout:

- `version1` – the original local-only workflow.
- `colab-version` – a notebook-friendly adaptation tailored for Google Colab.

Create feature branches from whichever base you are iterating on and keep them in sync
with `git pull` before committing. Switch branches with `git checkout <branch-name>`.

## 📦 Dataset

Download **Game_of_Thrones_Script.csv** from Kaggle and place it in the repository before
running any scripts:

- Source: [Game of Thrones Script (All Seasons) on Kaggle](https://www.kaggle.com/datasets/abhi3868/game-of-thrones-script-all-seasons)
- Local path: `data/Game_of_Thrones_Script.csv`

### Branch-specific placement

- **`version1`** – the legacy scripts load the file from `data/Game_of_Thrones_Script.csv`
  relative to the project root. Ensure the `data/` directory exists and copy the CSV
  there before executing helper scripts.
- **`colab-version`** – after cloning the repository in Google Colab (e.g., under
  `/content/MLBA-project`), upload the dataset to `/content/MLBA-project/data/` so the
  notebooks and CLI calls resolve the same relative path.

### Required schema & encoding

- The CSV must expose a character column (`Character`, `Name`, `Speaker`, or `Who`) and a
  dialogue column (`Dialogue`, `Sentence`, `Line`, or `Text`). The preprocessing utilities
  standardize these to the canonical `Character` and `Dialogue` headers.
- Save the file with UTF-8 encoding to avoid Unicode decoding issues on non-Unix hosts.
- Remove any empty trailing rows so the loaders can drop `NaN` entries cleanly.

### Cleanup & licensing notes

- No manual text normalization is required beyond verifying that each row has a character
  and line of dialogue—the helper scripts trim whitespace automatically.
- Respect the license posted on the Kaggle dataset page and include attribution to the
  dataset author when redistributing or publishing generated results.

## 🧠 Training & Generation

Choose whichever entry point fits your workflow:

### Single-command launcher
```bash
# Train, generate, analyze, or plot dashboards from one script
./run_project.sh --help
```
The launcher automatically exports `PYTHONPATH=src`, installs dependencies inside a
virtual environment, and invokes the appropriate module with `python -m`.

### Direct module execution
```bash
# 200-epoch full training run (heavy)
python -m got_script_generator.main_modern

# Enhanced training recipe with additional sampling callbacks
python -m got_script_generator.modern_example_usage

# Generate dashboards from training logs
python -m got_script_generator.modern_plot
```
> These commands start full training loops; adjust configuration inside the modules or
> use the `--quick` flag in `run_project.sh` for shorter experiments.

### Handy data-prep and validation scripts
```bash
python scripts/check_data.py
python scripts/test_columns.py
python scripts/quick_generation_test.py
```
Each script now imports the packaged modules via the `src/` layout and can be executed
from the repository root.

## 📁 Project Structure

```
MLBA-project/
├── README.md
├── requirements.txt
├── src/
│   └── got_script_generator/
│       ├── __init__.py
│       ├── cli.py
│       ├── improved_helperAI.py
│       ├── main_modern.py
│       ├── modern_example_usage.py
│       └── modern_plot.py
├── scripts/
│   ├── check_data.py
│   ├── debug.py
│   ├── quick_generation_test.py
│   ├── test_columns.py
│   ├── test_improvements.py
│   ├── test_modern.py
│   ├── test_pipe_fix.py
│   └── validate_pipe_fix.py
├── archive/                      # Historical reports & documentation
├── data/Game_of_Thrones_Script.csv
├── run_project.sh                # Cross-platform launcher
├── run_project.bat
└── run_project_optimized.sh
```

## 📚 Additional Resources

- `archive/` — in-depth reports on bug fixes, training improvements, and user guides.
- `scripts/` — targeted validation utilities and debugging helpers.
- `data/` — the curated Game of Thrones dialogue dataset used for training.

## 🛡️ License

This project is released under the permissive [MIT License](LICENSE). The MIT terms
cover the code in this repository, enabling reuse in commercial and open-source
settings with attribution.

Third-party assets bundled here, such as `data/Game_of_Thrones_Script.csv`, are
compiled by Abhijeet Anand (Kaggle user `abhi3868`). They remain subject to the
license and attribution requirements published on the
[Kaggle dataset page](https://www.kaggle.com/datasets/abhi3868/game-of-thrones-script-all-seasons).
Ensure your usage of the dataset complies with those terms—in particular, retain
attribution to the original compiler when sharing derivatives—while the
repository’s source code continues to be governed by MIT.

## 🛠️ Troubleshooting

- **Import errors** – ensure you ran `pip install -e .` so Python can locate the
  `got_script_generator` package, or export `PYTHONPATH=$(pwd)/src` before running
  scripts.
- **CUDA issues** – use the `--cpu` flag in `run_project.sh` or install CPU-only
  versions of PyTorch from the official wheel index.
- **Missing dataset** – verify that `data/Game_of_Thrones_Script.csv` exists or provide
  the `--data` flag to CLI commands with your dataset path.

Happy scripting, and may your model sit upon the Iron Throne of dialogue generation! 🐉⚔️👑
