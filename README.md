# 🎬 Modern TV Script Generation

An advanced **neural network powered text generation** project designed to create TV script dialogue using modern NLP techniques.  
This project features:
- Multi‑format **dataset loading** (Excel, CSV, JSON, TXT)
- **Advanced tokenization** (BPE, GPT‑2, BERT, SentencePiece)
- **Character‑aware preprocessing** (speaker tags for better context)
- **Bidirectional LSTM** with **attention mechanisms** and **character embeddings**
- **Modern training** features (AdamW optimizer, learning rate scheduling, mixed precision)
- **Advanced text generation sampling** (nucleus sampling, temperature scaling)
- Robust **logging, saving/loading models, and plotting training metrics**

---

## 📦 Features
- 📥 **Load scripts** from multiple formats (CSV/XLSX/JSON/TXT)
- 🧠 Train modern RNN/Transformer-like text models
- 🎭 Generate **character-specific** lines in the style of your dataset
- 📊 Visualize training with Plotly (`loss` & `learning rate`)
- 🔄 Save & resume training seamlessly

---

## 🛠 Requirements
- **Python 3.10 or 3.11 recommended** (avoid 3.13 until torch wheels support it)
- Modern NVIDIA GPU (GTX/RTX) **or** CPU
- Compatible NVIDIA drivers:
  - CUDA 12.1 runtime recommended
  - CUDA 11.8 runtime for older setups

---

## 🔧 Installation

Clone this repository:

```sh
git clone https://github.com/yourusername/tv-script-generation.git
cd tv-script-generation
```

Optional but **recommended**: Create a virtual environment:

```sh
python -m venv venv
.\venv\Scripts\Activate
```

Install dependencies from `requirements.txt`:

```sh
pip install -U pip
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install essentials manually:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets sentencepiece pandas numpy plotly kaleido pytest
```

---

## 💻 Running the Project (Windows PowerShell)

Here’s an **easy step-by-step** guide for **beginners**:

### **1️⃣ Open PowerShell**
- Press **Start**, search for `PowerShell` or `Windows Terminal` (PowerShell tab)
- Change to your project folder:

```sh
cd "C:\Path\To\project-tv-script-generation"
```

### **2️⃣ Create & Activate a Virtual Environment**

```sh
python -m venv venv
.\venv\Scripts\Activate
```

You should see `(venv)` before your PowerShell prompt.

### **3️⃣ Install packages**

```sh
pip install -U pip
pip install -r requirements.txt
```

### **4️⃣ Verify Python & CUDA**

```sh
python --version
python -c "import torch; print(torch.version); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: True` means GPU is ready.

### **5️⃣ Preprocess Dataset**
Ensure your dataset is in `data/` and matches the `DATA_PATH` in `main_modern.py`.

To check dataset stats:

```sh
python -c "from improved_helperAI import analyze_dataset; print(analyze_dataset('data/Game_of_Thrones_Script.csv'))"
```

### **6️⃣ Train & Generate Dialogue**

```sh
python main_modern.py
```

This will:
- Preprocess data → `preprocess_modern.pkl`
- Train model → save to `modern_script_model.pt`
- Generate dialogues for main characters

### **7️⃣ Plot Training Metrics (Optional)**

```sh
python -c "from modern_plot import parse_log_file, plot_training_metrics; recs = parse_log_file('train.log'); plot_training_metrics(recs, metrics=['loss','lr'])"
```

### **8️⃣ Generate Without Retraining**

```sh
python -c "from modern_example_usage import ModernGenerator, ModernScriptRNN; from improved_helperAI import load_modern_model, load_modern_preprocess, ModernTextProcessor; pre = load_modern_preprocess('preprocess_modern.pkl'); model, _ = load_modern_model('modern_script_model', ModernScriptRNN, vocab_size=pre['metadata']['vocab_size'], output_size=pre['metadata']['vocab_size'], embedding_dim=256, hidden_dim=512, n_layers=2, characters=pre['metadata']['characters']); gen = ModernGenerator(model, pre['vocab_to_int'], pre['int_to_vocab'], pre['character_vocab'], tokenizer=ModernTextProcessor('gpt2', 'gpt2').tokenizer); print(gen.generate_nucleus_sampling('jon snow:', 100, character='jon snow'))"
```

---

## 📂 Project Structure

1. 📁 data/ # Your datasets
2. 📄 main_modern.py # Entrypoint for training/generation
3. 📄 improved_helperAI.py # Preprocessing, loading, saving
4. 📄 modern_example_usage.py # Model architecture, training, generation logic
5. 📄 modern_plot.py # Plotting helper functions
6. 📄 test_modern.py # Unit tests
7. 📄 requirements.txt # Dependencies

---

## 📝 Notes
- You can update `DATA_PATH` in `main_modern.py` to point to your dataset
- Larger datasets require **more VRAM**; adjust `BATCH_SIZE` and `CONTEXT_WINDOW` if you get `CUDA out-of-memory`
- Training log is saved in `training.log` if enabled
- `.gitignore` should be configured to avoid committing large model/data files

---

## 📜 License
This project is free to use for personal and educational purposes.  
For commercial use, please seek permission from the author.

---

## 🙌 Acknowledgements
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- Game of Thrones dataset inspiration from **Udacity Deep Learning Nanodegree**.

---

