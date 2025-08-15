# 🎬 Game of Thrones AI Script Generator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **advanced neural network** that learns to write Game of Thrones dialogue! 🐉⚔️👑

This AI system reads thousands of lines from the show and learns to generate new dialogue that sounds like it could be from the actual series. Perfect for fans, writers, and AI enthusiasts!

## ✨ Features

🎭 **Character-Aware Generation** - Writes dialogue specific to Jon Snow, Tyrion, Daenerys, etc.  
📊 **Advanced Training** - 200 epochs with comprehensive logging and progress tracking  
📈 **Real-time Monitoring** - Watch your AI learn with detailed progress reports  
🎨 **Interactive Visualizations** - Beautiful charts showing training progress  
💾 **Smart Checkpointing** - Never lose training progress with auto-saves  
🚀 **GPU Accelerated** - Fast training on modern NVIDIA cards  
📝 **Detailed Logging** - Complete training logs saved to `training_output.txt`  

---

## 🚀 Quick Start Commands

### **🏁 One-Click Training (Recommended)**
```powershell
python main_modern.py
```
*Trains for 200 epochs with full logging and visualization*

### **⚡ Extended Training (Best Quality)**
```powershell
python modern_example_usage.py
```
*Enhanced 200-epoch training with sample generation every 5 epochs*

### **📊 Create Training Dashboard**
```powershell
python -c "from modern_plot import quick_dashboard; quick_dashboard()"
```
*Generates interactive HTML visualizations of training progress*

### **🎭 Generate Dialogue (Quick Test)**
```powershell
python -c "from modern_example_usage import quick_generate; quick_generate('jon snow:', character='jon snow')"
```

### **📈 Analyze Dataset**
```powershell
python -c "from improved_helperAI import analyze_dataset; print(analyze_dataset('data/Game_of_Thrones_Script.csv'))"
```

---

## 🛠️ Installation

### **Prerequisites**
- **Python 3.10 or 3.11** (recommended)
- **NVIDIA GPU** (optional but recommended for faster training)
- **8GB+ RAM** (16GB+ recommended)

### **Step 1: Clone Repository**
```powershell
git clone https://github.com/saint2706/MLBA-project.git
cd MLBA-project
```

### **Step 2: Create Virtual Environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### **Step 3: Install Dependencies**
```powershell
pip install -U pip
pip install -r requirements.txt
```

### **Step 4: Verify Installation**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 📊 Training Options

| Command | Duration | Quality | Best For |
|---------|----------|---------|----------|
| `python main_modern.py` | 4-8 hours | High | **Most users** |
| `python modern_example_usage.py` | 8+ hours | Highest | **Best results** |

### **🎛️ Customization Options**

**Change training duration:**
```powershell
python -c "exec(open('main_modern.py').read().replace('NUM_EPOCHS = 200', 'NUM_EPOCHS = 50'))"
```

**GPU memory optimization (if you get CUDA errors):**
```powershell
python -c "exec(open('main_modern.py').read().replace('BATCH_SIZE = 16', 'BATCH_SIZE = 8'))"
```

---

## 🎭 Generation Commands

### **Generate with Different Characters**
```powershell
# Jon Snow
python -c "from modern_example_usage import quick_generate; quick_generate('jon snow: ', character='jon snow')"

# Tyrion Lannister  
python -c "from modern_example_usage import quick_generate; quick_generate('tyrion: ', character='tyrion')"

# Daenerys Targaryen
python -c "from modern_example_usage import quick_generate; quick_generate('daenerys: ', character='daenerys')"
```

### **Control Creativity Level**
```powershell
# Conservative (more predictable)
python -c "from modern_example_usage import quick_generate; quick_generate('jon snow: ', temperature=0.5)"

# Creative (more surprising)  
python -c "from modern_example_usage import quick_generate; quick_generate('jon snow: ', temperature=1.2)"
```

### **Generate Longer Dialogue**
```powershell
python -c "from modern_example_usage import quick_generate; quick_generate('jon snow: ', max_length=300)"
```

---

## 📊 Monitoring & Analysis

### **📈 Real-time Training Progress**
```powershell
# Watch training logs live
Get-Content training_output.txt -Wait -Tail 10
```

### **📊 Generate All Visualizations**
```powershell
python modern_plot.py
```

### **🔍 Check Model Performance**
```powershell
python -c "from modern_plot import parse_training_log; metrics = parse_training_log('training_output.txt'); print(f'Best Loss: {min(metrics[\"loss\"]):.4f}')"
```

### **📋 Training Summary**
```powershell
python -c "import os; print(f'Model files: {[f for f in os.listdir() if f.endswith(\".pt\")]}'); print(f'Log size: {os.path.getsize(\"training_output.txt\") if os.path.exists(\"training_output.txt\") else 0} bytes')"
```

---

## 📁 Project Structure

```
project-tv-script-generation/
├── 🎬 main_modern.py              # 👈 START HERE - Main training script
├── 🚀 modern_example_usage.py     # Enhanced 200-epoch training
├── 🤖 improved_helperAI.py        # Core AI helper functions  
├── 📊 modern_plot.py              # Visualization tools
├── 📚 NON_PROGRAMMER_GUIDE.md     # Complete user guide
├── 📋 CODE_QUALITY_ENHANCEMENTS.md # Technical improvements
├── 🔧 requirements.txt            # Python dependencies
├── 📝 training_output.txt         # Training logs (auto-generated)
├── 🧠 *.pt files                  # Trained models (auto-generated)
├── 📊 *.html files                # Interactive visualizations
└── 📁 data/                       # Your training data
    └── Game_of_Thrones_Script.csv
```

---

## 🎯 Usage Examples

### **🏃‍♂️ Quick Training Session (1-2 hours)**
```powershell
# Modify for shorter training
python -c "
import main_modern
main_modern.NUM_EPOCHS = 50  # Reduce epochs
exec(open('main_modern.py').read())
"
```

### **🎭 Generate Multiple Character Samples**
```powershell
python -c "
characters = ['jon snow', 'tyrion', 'daenerys', 'arya', 'cersei']
from modern_example_usage import quick_generate
for char in characters:
    print(f'\n=== {char.upper()} ===')
    print(quick_generate(f'{char}: ', character=char, max_length=100))
"
```

### **📊 Complete Analysis Pipeline**
```powershell
# 1. Analyze data → 2. Train → 3. Visualize → 4. Generate
python -c "from improved_helperAI import analyze_dataset; print(analyze_dataset('data/Game_of_Thrones_Script.csv'))" && python main_modern.py && python modern_plot.py && python -c "from modern_example_usage import quick_generate; print(quick_generate('jon snow: ', character='jon snow'))"
```

---

## 🔧 Troubleshooting

### **🚨 Common Issues & Solutions**

| Problem | Solution Command |
|---------|------------------|
| CUDA out of memory | `python -c "exec(open('main_modern.py').read().replace('BATCH_SIZE = 16', 'BATCH_SIZE = 4'))"` |
| Training too slow | `python -c "exec(open('main_modern.py').read().replace('NUM_EPOCHS = 200', 'NUM_EPOCHS = 50'))"` |
| Can't find data file | `ls data/` then update `DATA_PATH` in `main_modern.py` |
| Import errors | `pip install -r requirements.txt` |

### **📝 Check System Status**
```powershell
python -c "
import torch, sys, os
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}') 
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'Data exists: {os.path.exists(\"data/Game_of_Thrones_Script.csv\")}')
"
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[NON_PROGRAMMER_GUIDE.md](NON_PROGRAMMER_GUIDE.md)** | Complete beginner's guide |
| **[CODE_QUALITY_ENHANCEMENTS.md](CODE_QUALITY_ENHANCEMENTS.md)** | Technical improvements list |
| **[TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md)** | Training system enhancements |

---

## 🎊 Success Indicators

**✅ Training is working when you see:**
- Decreasing loss values over time
- Regular checkpoint saves every 10 epochs  
- Sample generation every 20 epochs showing improvement
- ETA calculations showing reasonable completion times

**🎭 Good dialogue output looks like:**
- Character-appropriate language and vocabulary
- Coherent sentence structure
- Contextually relevant responses
- Proper Game of Thrones terminology and themes

---

## 💡 Pro Tips

### **🚀 Performance Optimization**
```powershell
# Use multiple workers for faster data loading
python -c "exec(open('main_modern.py').read().replace('num_workers=2', 'num_workers=4'))"
```

### **🎯 Fine-tune Generation**
```powershell
# More focused dialogue
python -c "from modern_example_usage import quick_generate; quick_generate('the night king: ', top_p=0.7, temperature=0.6)"

# More creative dialogue  
python -c "from modern_example_usage import quick_generate; quick_generate('tyrion: ', top_p=0.95, temperature=1.1)"
```

### **📊 Advanced Monitoring**
```powershell
# Monitor GPU usage during training
nvidia-smi -l 5
```

---

## 🤝 Contributing

Found a bug or want to improve the project?
1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Game of Thrones** dataset from Kaggle
- **PyTorch** team for the deep learning framework
- **Hugging Face** for transformer models and tokenizers
- **Plotly** for interactive visualizations
- The **open source community** for inspiration and tools

---

## 🎉 Have Fun!

**Winter is coming... but your AI is ready!** 🐺❄️

Generate epic dialogue, experiment with different characters, and watch your AI learn the art of Westerosi conversation! 

**May your training losses be ever in your favor!** ⚔️🏰👑
