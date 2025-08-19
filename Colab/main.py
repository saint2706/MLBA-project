# -*- coding: utf-8 -*-
"""
Game of Thrones AI Script Generator (Colab Edition)

This script is a self-contained, Colab-friendly version of the project,
designed to be run cell-by-cell in a Google Colab notebook.

Instructions for Colab:
1. Upload this file to your Google Drive.
2. Open it with Google Colaboratory.
3. Upload the `Game_of_Thrones_Script.csv` dataset to the same directory
   in your Colab environment (or modify the `DATA_PATH` variable).
4. Ensure your Colab runtime is set to use a GPU for faster training
   (Runtime -> Change runtime type -> GPU).
5. Run the cells in order.
"""

# ================================================================
# 📝 STEP 1: SETUP AND DEPENDENCY INSTALLATION
# ================================================================
# This cell installs all the necessary Python libraries.

import subprocess
import sys

def install_packages():
    """Installs required packages using pip."""
    packages = [
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "torchaudio>=2.4.0",
        "transformers>=4.44.0",
        "datasets>=2.20.0",
        "tokenizers>=0.19.1",
        "sentencepiece>=0.2.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "scikit-learn>=1.5.0",
        "plotly>=5.22.0",
        "kaleido>=0.2.1",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
        "openpyxl>=3.1.0",
        "accelerate>=0.33.0",
        "psutil>=6.0.0"
    ]
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

# Uncomment the line below to run the installation.
# It's recommended to run this only once per session.
# install_packages()


# ================================================================
# 📦 STEP 2: IMPORT LIBRARIES
# ================================================================
# All necessary imports are gathered here.

import os
import pickle
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from collections import Counter
import datetime
import re

try:
    from transformers import GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers library not available. Using basic tokenization.")


# ================================================================
# ⚙️ STEP 3: HELPER FUNCTIONS AND CLASSES (FROM HELPERAI & EXAMPLES)
# ================================================================
# This section contains all the consolidated classes and functions
# from the original project files.

# --- Global Settings & Special Tokens ---

SPECIAL_WORDS = {
    "PADDING": "<PAD>", "UNKNOWN": "<UNK>", "START": "<START>", "END": "<END>",
    "SPEAKER_SEP": "<SSEP>", "SCENE_SEP": "<SCESEP>",
}

# --- Data Loading and Preprocessing Functions ---

def load_structured_data(path: Union[str, Path]) -> pd.DataFrame:
    """Loads dialogue data from CSV, Excel, or JSON files."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Standardize column names
        char_cols = ["name", "character", "speaker", "who", "Name"]
        dialogue_cols = ["sentence", "dialogue", "text", "line", "speech", "Sentence"]

        for col in df.columns:
            if col.lower() in char_cols:
                df = df.rename(columns={col: "Character"})
                break
        for col in df.columns:
            if col.lower() in dialogue_cols:
                df = df.rename(columns={col: "Dialogue"})
                break

        if "Character" not in df.columns or "Dialogue" not in df.columns:
            raise ValueError(f"Dataset must have Character and Dialogue columns. Found: {df.columns.tolist()}")

        df = df[["Character", "Dialogue"]].dropna()
        df["Character"] = df["Character"].astype(str).str.lower().str.strip()
        df["Dialogue"] = df["Dialogue"].astype(str).str.strip()
        print(f"✅ Loaded {len(df)} dialogue entries from {path}")
        return df
    except Exception as e:
        print(f"❌ Error loading data from {path}: {e}")
        raise

def analyze_dataset(data_path: Union[str, Path]) -> Dict:
    """Analyzes the dataset and returns key statistics."""
    df = load_structured_data(data_path)
    analysis = {
        "total_dialogues": len(df),
        "unique_characters": df["Character"].nunique(),
        "character_counts": df["Character"].value_counts().to_dict(),
        "avg_dialogue_length": df["Dialogue"].str.len().mean(),
        "total_words": df["Dialogue"].str.split().str.len().sum(),
        "vocabulary_size": len(set(" ".join(df["Dialogue"]).lower().split())),
    }
    return analysis

class ModernTextProcessor:
    """Handles text tokenization and processing."""
    def __init__(self, tokenizer_type: str = "gpt2", model_name: str = "gpt2"):
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE and self.tokenizer_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer_type = "basic"
            print("Falling back to basic word splitting for tokenization.")

def create_lookup_tables(text_data: List[str], min_freq: int, max_vocab: int) -> Tuple[Dict, Dict]:
    """Creates vocabulary-to-integer mapping and vice-versa."""
    word_counts = Counter(" ".join(text_data).lower().split())
    sorted_vocab = sorted([w for w, c in word_counts.items() if c >= min_freq], key=word_counts.get, reverse=True)
    if max_vocab:
        sorted_vocab = sorted_vocab[:max_vocab - len(SPECIAL_WORDS)]
    final_vocab = list(SPECIAL_WORDS.values()) + sorted_vocab
    vocab_to_int = {word: i for i, word in enumerate(final_vocab)}
    int_to_vocab = {i: word for i, word in enumerate(final_vocab)}
    return vocab_to_int, int_to_vocab

def preprocess_and_save_data(data_path, output_path, tokenizer_type, model_name, min_frequency, max_vocab_size, context_window):
    """Main function to preprocess data and save it to a file."""
    df = load_structured_data(data_path)
    processor = ModernTextProcessor(tokenizer_type, model_name)

    char_vocab = {c: f"<{c.upper()}>" for c in df["Character"].unique()}
    formatted_text = [f"{char_vocab.get(row['Character'], '<UNKNOWN>')} {row['Dialogue']}" for _, row in df.iterrows()]
    full_text = " ".join(formatted_text)
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    if processor.tokenizer:
        tokens = processor.tokenizer.encode(full_text, add_special_tokens=True)
        vocab_to_int = {tok: i for tok, i in processor.tokenizer.get_vocab().items()}
        int_to_vocab = {i: tok for tok, i in vocab_to_int.items()}
    else:
        text_words = full_text.lower().split()
        vocab_to_int, int_to_vocab = create_lookup_tables(text_words, min_frequency, max_vocab_size)
        tokens = [vocab_to_int.get(w, vocab_to_int[SPECIAL_WORDS["UNKNOWN"]]) for w in text_words]

    sequences = []
    for i in range(0, len(tokens) - context_window, context_window // 2):
        seq = tokens[i:i + context_window]
        if len(seq) == context_window:
            sequences.append(seq)

    metadata = {
        "vocab_size": len(vocab_to_int), "context_window": context_window,
        "num_sequences": len(sequences), "characters": list(char_vocab.keys()),
        "character_vocab": char_vocab,
    }

    preprocessed_data = {
        "sequences": sequences, "vocab_to_int": vocab_to_int,
        "int_to_vocab": int_to_vocab, "metadata": metadata,
    }

    with open(output_path, "wb") as f:
        pickle.dump(preprocessed_data, f)

    print(f"✅ Preprocessing complete. Data saved to {output_path}")
    return preprocessed_data

def load_preprocessed_data(file_path):
    """Loads preprocessed data from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

# --- Model Saving and Loading ---

def save_model(filepath, model, optimizer=None, metadata=None):
    """Saves the model state and metadata."""
    state = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, filepath + ".pt")
    print(f"💾 Model saved to {filepath}.pt")

def load_model(filepath, model_class, **model_kwargs):
    """Loads a model from a checkpoint."""
    checkpoint = torch.load(filepath + ".pt", map_location="cpu")
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint.get("metadata", {})

# --- AI Model and Trainer Classes ---

def create_character_embeddings(characters: List[str], embedding_dim: int) -> nn.Embedding:
    """Creates an embedding layer for characters."""
    return nn.Embedding(len(characters), embedding_dim)

class ModernScriptRNN(nn.Module):
    """The main RNN model with Attention for script generation."""
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, characters, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.char_embedding = create_character_embeddings(characters, embedding_dim // 4)
        self.lstm = nn.LSTM(embedding_dim + embedding_dim // 4, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, input_seq, hidden, char_ids=None):
        batch_size, seq_len = input_seq.size()
        word_embeds = self.embedding(input_seq)
        if char_ids is not None:
            char_embeds = self.char_embedding(char_ids).unsqueeze(1).expand(-1, seq_len, -1)
            embeds = torch.cat([word_embeds, char_embeds], dim=-1)
        else:
            char_pad = torch.zeros(batch_size, seq_len, self.char_embedding.embedding_dim, device=input_seq.device)
            embeds = torch.cat([word_embeds, char_pad], dim=-1)

        lstm_out, hidden = self.lstm(embeds, hidden)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(device))

class ModernTrainer:
    """Handles the model training loop, optimization, and logging."""
    def __init__(self, model, train_loader, vocab_to_int, int_to_vocab):
        self.model = model
        self.train_loader = train_loader
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab_to_int.get(SPECIAL_WORDS["PADDING"], 0))
        self.logger = logging.getLogger('GoT_AI_Trainer')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler = logging.FileHandler('training_output.txt', mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def train_epoch(self, epoch_num):
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_loader:
            device = next(self.model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = self.model.init_hidden(inputs.size(0), device)

            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs, hidden)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch_num} | Avg Loss: {avg_loss:.6f}")
        return avg_loss

class ModernGenerator:
    """Handles text generation using the trained model."""
    def __init__(self, model, vocab_to_int, int_to_vocab, char_vocab, tokenizer):
        self.model = model
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        self.char_vocab = char_vocab
        self.tokenizer = tokenizer

    def generate(self, seed_text, max_len, temp, top_p, char, penalty):
        self.model.eval()
        device = next(self.model.parameters()).device

        tokens = self.tokenizer.encode(seed_text) if self.tokenizer else \
                 [self.vocab_to_int.get(w, 0) for w in seed_text.lower().split()]

        generated_tokens = []
        for _ in range(max_len):
            input_ids = torch.tensor([tokens[-128:]], device=device) # Use a sliding window
            hidden = self.model.init_hidden(1, device)
            char_id = torch.tensor([list(self.char_vocab.keys()).index(char)]).to(device) if char else None

            with torch.no_grad():
                logits, _ = self.model(input_ids, hidden, char_id)

            logits = logits[0, -1, :] / temp
            if penalty != 1.0 and generated_tokens:
                for token_id in set(generated_tokens):
                    logits[token_id] /= penalty if logits[token_id] > 0 else penalty

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            indices_to_remove = cum_probs > top_p
            indices_to_remove[1:] = indices_to_remove[:-1].clone()
            indices_to_remove[0] = False
            logits[sorted_indices[indices_to_remove]] = -float('Inf')

            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            generated_tokens.append(next_token)
            tokens.append(next_token)

        return self.tokenizer.decode(generated_tokens) if self.tokenizer else \
               " ".join([self.int_to_vocab.get(t, "") for t in generated_tokens])


# ================================================================
# 🚀 STEP 4: MAIN EXECUTION SCRIPT
# ================================================================
if __name__ == '__main__':
    # --- Configuration ---
    DATA_PATH = "Game_of_Thrones_Script.csv"
    PREPROCESS_PATH = "colab_preprocess.pkl"
    MODEL_SAVE_PATH = "colab_got_model"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {DEVICE}")

    # --- Text Processing & Training Hyperparameters ---
    TOKENIZER_TYPE = "gpt2"
    MODEL_NAME = "gpt2"
    MIN_FREQUENCY = 3
    MAX_VOCAB_SIZE = 15000
    CONTEXT_WINDOW = 128
    BATCH_SIZE = 16
    NUM_EPOCHS = 50 # Reduced for faster demo in Colab
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3

    # --- 1. Data Analysis ---
    print("\n🔍 Analyzing dataset...")
    analysis = analyze_dataset(DATA_PATH)
    print(f"   Total dialogues: {analysis['total_dialogues']}")
    print(f"   Unique characters: {analysis['unique_characters']}")

    # --- 2. Data Preprocessing ---
    print("\n📝 Preprocessing data...")
    preprocessed_data = preprocess_and_save_data(
        DATA_PATH, PREPROCESS_PATH, TOKENIZER_TYPE, MODEL_NAME,
        MIN_FREQUENCY, MAX_VOCAB_SIZE, CONTEXT_WINDOW
    )
    metadata = preprocessed_data["metadata"]
    vocab_size = metadata["vocab_size"]
    characters = metadata["characters"]

    # --- 3. Prepare DataLoader ---
    print("\n🔧 Preparing data loaders...")
    sequences = np.array(preprocessed_data["sequences"])
    data_x = torch.tensor(sequences[:, :-1], dtype=torch.long)
    data_y = torch.tensor(sequences[:, 1:], dtype=torch.long)
    train_data = TensorDataset(data_x, data_y)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # --- 4. Initialize Model ---
    print("\n🧠 Initializing model...")
    model = ModernScriptRNN(
        vocab_size, vocab_size, EMBEDDING_DIM, HIDDEN_DIM,
        NUM_LAYERS, characters, DROPOUT
    ).to(DEVICE)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- 5. Train the Model ---
    print("\n🎓 Starting training...")
    trainer = ModernTrainer(model, train_loader, preprocessed_data["vocab_to_int"], preprocessed_data["int_to_vocab"])
    for epoch in range(1, NUM_EPOCHS + 1):
        trainer.train_epoch(epoch)
        if epoch % 10 == 0:
            save_model(f"{MODEL_SAVE_PATH}_epoch_{epoch}", model)

    save_model(MODEL_SAVE_PATH, model, trainer.optimizer, metadata)
    print("\n🎉 Training complete!")

    # --- 6. Generate Script ---
    print("\n🎭 Generating sample script...")
    processor = ModernTextProcessor(TOKENIZER_TYPE, MODEL_NAME)
    generator = ModernGenerator(
        model, preprocessed_data["vocab_to_int"], preprocessed_data["int_to_vocab"],
        metadata["character_vocab"], processor.tokenizer
    )

    sample_script = generator.generate(
        seed_text="<TYRION LANNISTER>", max_len=200, temp=0.8,
        top_p=0.9, char="tyrion lannister", penalty=1.2
    )
    print("\n--- Generated Script ---")
    print(sample_script)
    print("------------------------")
