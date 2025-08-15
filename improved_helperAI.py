# improved_helperAI.py  
# 🤖 ADVANCED AI HELPER FUNCTIONS FOR TV SCRIPT GENERATION
#
# FOR NON-PROGRAMMERS:
# This file contains all the "behind-the-scenes" functions that help our AI work.
# Think of it like a toolbox full of specialized tools that:
# 
# 1. 📖 READ AND UNDERSTAND DATA: Loads Game of Thrones scripts from files
# 2. 🔄 PREPARE TEXT FOR AI: Converts human text into numbers the AI can understand  
# 3. 🧠 MANAGE AI MODELS: Saves and loads our trained AI models
# 4. 📊 ANALYZE PATTERNS: Studies the data to understand character speech patterns
# 5. 🎯 CREATE TRAINING BATCHES: Organizes data for efficient AI learning
#
# You don't need to understand every line here - just know that these functions
# work together to make our AI script generator possible!
#
# Modern TV Script Generation Helper Functions
# Implements advanced preprocessing, tokenization, and model utilities  
# Updated for 2024 with Transformer support and modern NLP techniques

import os
import pickle
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from collections import Counter

# Advanced tokenization and preprocessing
# These libraries help us process text more intelligently
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BertTokenizer,
        BertModel,
        GPT2Tokenizer,
        GPT2LMHeadModel,
    )
    from datasets import Dataset as HFDataset

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("⚠️ Transformers library not available. Using basic tokenization.")

# Modern preprocessing libraries
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

# Configure logging (helps us track what the program is doing)
logging.basicConfig(
    level=logging.INFO,
    filename="training.log",
    filemode="w",
    format="%(asctime)s %(levelname)s:%(message)s",
)
logger = logging.getLogger(__name__)

# Enhanced special tokens for modern preprocessing
# These are special markers that help the AI understand text structure
SPECIAL_WORDS = {
    "PADDING": "<PAD>",
    "UNKNOWN": "<UNK>",
    "START": "<START>",
    "END": "<END>",
    "SPEAKER_SEP": "<SSEP>",
    "SCENE_SEP": "<SCESEP>",
}


class ModernTextProcessor:
    """
    Modern text preprocessing class with support for:
    - Advanced tokenization (BPE, SentencePiece)
    - Contextual embeddings (BERT, GPT-2)
    - Character-aware preprocessing for TV scripts
    - Multi-format data loading (Excel, CSV, JSON, TXT)
    """

    def __init__(self, tokenizer_type: str = "bpe", model_name: str = "gpt2"):
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self.tokenizer = None
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.character_vocab = set()
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize modern tokenizer"""
        if not TRANSFORMERS_AVAILABLE and self.tokenizer_type != "basic":
            logger.warning(
                "Transformers not available, falling back to basic tokenization"
            )
            self.tokenizer_type = "basic"
            return

        if self.tokenizer_type == "gpt2" and TRANSFORMERS_AVAILABLE:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.tokenizer_type == "bert" and TRANSFORMERS_AVAILABLE:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.tokenizer_type == "bpe" and TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(
            f"Initialized {self.tokenizer_type} tokenizer with {self.model_name}"
        )


def load_structured_data(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        elif path.suffix.lower() == ".txt":
            return load_legacy_text_data(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        df = standardize_dataframe_columns(df)
        logger.info(f"Loaded {len(df)} dialogue entries from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {path}: {str(e)}")
        return load_legacy_text_data(path)


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    character_columns = ["name", "character", "speaker", "who"]
    dialogue_columns = ["sentence", "dialogue", "text", "line", "speech"]

    for col in df.columns:
        if col.lower() in character_columns:
            df = df.rename(columns={col: "Character"})
            break
    for col in df.columns:
        if col.lower() in dialogue_columns:
            df = df.rename(columns={col: "Dialogue"})
            break

    if "Character" not in df.columns or "Dialogue" not in df.columns:
        raise ValueError("Dataset must contain character and dialogue columns")

    df = df[["Character", "Dialogue"]].dropna()
    df["Character"] = df["Character"].astype(str).str.lower().str.strip()
    df["Dialogue"] = df["Dialogue"].astype(str).str.strip()
    return df


def load_legacy_text_data(path: Union[str, Path]) -> pd.DataFrame:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            text = f.read()

    if len(text) > 81:
        text = text[81:]

    lines = text.strip().split("\n")
    characters, dialogues = [], []
    for line in lines:
        line = line.strip()
        if ":" in line and line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                character = parts[0].strip().lower()
                dialogue = parts[1].strip()
                if character and dialogue:
                    characters.append(character)
                    dialogues.append(dialogue)
    if not characters:
        dialogues = [line.strip() for line in lines if line.strip()]
        characters = ["unknown"] * len(dialogues)

    df = pd.DataFrame({"Character": characters, "Dialogue": dialogues})
    logger.info(f"Parsed {len(df)} dialogue entries from legacy text format")
    return df


def create_modern_lookup_tables(
    text_data: List[str], min_frequency: int = 2, max_vocab_size: Optional[int] = None
) -> Tuple[Dict, Dict]:
    word_counts = Counter()
    for text in text_data:
        words = text.lower().split()
        word_counts.update(words)
    filtered_words = [
        word for word, count in word_counts.items() if count >= min_frequency
    ]
    sorted_vocab = sorted(filtered_words, key=word_counts.get, reverse=True) # type: ignore
    if max_vocab_size:
        sorted_vocab = sorted_vocab[: max_vocab_size - len(SPECIAL_WORDS)]
    final_vocab = list(SPECIAL_WORDS.values()) + sorted_vocab
    vocab_to_int = {word: idx for idx, word in enumerate(final_vocab)}
    int_to_vocab = {idx: word for idx, word in enumerate(final_vocab)}
    logger.info(f"Created vocabulary with {len(vocab_to_int)} tokens")
    return vocab_to_int, int_to_vocab


def advanced_token_lookup() -> Dict[str, str]:
    return {
        ".": "||Period||",
        ",": "||Comma||",
        "!": "||Exclamation||",
        "?": "||Question||",
        ";": "||Semicolon||",
        ":": "||Colon||",
        '"': "||Quote||",
        "'": "||Apostrophe||",
        "(": "||LParen||",
        ")": "||RParen||",
        "[": "||LBracket||",
        "]": "||RBracket||",
        "-": "||Dash||",
        "–": "||NDash||",
        "—": "||MDash||",
        "\n": "||NewLine||",
        "\t": "||Tab||",
        "...": "||Ellipsis||",
        "/": "||Slash||",
        "&": "||Ampersand||",
        "@": "||At||",
        "#": "||Hash||",
        "$": "||Dollar||",
        "%": "||Percent||",
        "*": "||Star||",
        "+": "||Plus||",
        "=": "||Equal||",
        "<": "||Less||",
        ">": "||Greater||",
        "^": "||Caret||",
        "_": "||Underscore||",
        "|": "||Pipe||",
        "~": "||Tilde||",
        "`": "||Backtick||",
    }


def preprocess_and_save_modern_data(
    data_path: Union[str, Path],
    output_path: str = "modern_preprocess.pkl",
    tokenizer_type: str = "gpt2",
    model_name: str = "gpt2",
    min_frequency: int = 2,
    max_vocab_size: Optional[int] = None,
    context_window: int = 1024,
    stride: int = 512,
):
    df = load_structured_data(data_path)
    processor = ModernTextProcessor(tokenizer_type, model_name)
    character_vocab = {c: f"<{c.upper()}>" for c in df["Character"].unique()}
    formatted = [
        f"{character_vocab.get(row['Character'], SPECIAL_WORDS['UNKNOWN'])} {row['Dialogue']}"
        for _, row in df.iterrows()
    ]

    all_text = " ".join(formatted)

    if processor.tokenizer is None:
        tok_map = advanced_token_lookup()
        for sym, token in tok_map.items():
            all_text = all_text.replace(sym, f" {token} ")
        text_words = all_text.lower().split()
        vocab_to_int, int_to_vocab = create_modern_lookup_tables(
            [" ".join(text_words)], min_frequency, max_vocab_size
        )
        token_dict = tok_map
        tokens = [
            vocab_to_int.get(w, vocab_to_int[SPECIAL_WORDS["UNKNOWN"]])
            for w in text_words
        ]
    else:
        token_dict = {}
        tokens = processor.tokenizer.encode(all_text, add_special_tokens=True)
        vocab_to_int = {
            tok: idx for tok, idx in processor.tokenizer.get_vocab().items()
        }
        int_to_vocab = {idx: tok for tok, idx in vocab_to_int.items()}

    # Enforce sliding-window with proper window and stride
    sequences = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + context_window]
        if len(chunk) < context_window:
            pad_id = (
                processor.tokenizer.pad_token_id
                if processor.tokenizer
                else vocab_to_int.get(SPECIAL_WORDS["PADDING"], 0)
            )
            chunk += [pad_id] * (context_window - len(chunk))
        sequences.append(chunk)

    metadata = {
        "tokenizer_type": tokenizer_type,
        "model_name": model_name,
        "vocab_size": len(vocab_to_int),
        "window": context_window,
        "stride": stride,
        "num_sequences": len(sequences),
        "characters": list(character_vocab.keys()),
        "character_vocab": character_vocab,
        "min_frequency": min_frequency,
        "max_vocab_size": max_vocab_size,
    }

    preprocessed_data = {
        "sequences": sequences,
        "vocab_to_int": vocab_to_int,
        "int_to_vocab": int_to_vocab,
        "token_dict": token_dict,
        "character_vocab": character_vocab,
        "metadata": metadata,
        "raw_data": df.to_dict("records"),
    }

    with open(output_path, "wb") as f:
        pickle.dump(preprocessed_data, f)

    logger.info(
        f"Saved preprocessed data with {len(sequences)} sequences to {output_path}"
    )

    return preprocessed_data


def create_contextual_sequences(
    encoded_text: List[int], context_window: int
) -> List[List[int]]:
    sequences = []
    step_size = context_window // 2
    for i in range(0, len(encoded_text) - context_window, step_size):
        sequence = encoded_text[i : i + context_window]
        if len(sequence) == context_window:
            sequences.append(sequence)
    return sequences


def load_modern_preprocess(file_path: str = "modern_preprocess.pkl") -> Dict:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    required_keys = ["sequences", "vocab_to_int", "int_to_vocab", "metadata"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in preprocessed data: {missing_keys}")
    logger.info(f"Loaded preprocessed data from {file_path}")
    return data


def save_modern_model(filepath: str, model, optimizer=None, metadata=None):
    save_data = {
        "model_state_dict": model.state_dict(),
        "model_config": getattr(model, "config", None),
        "timestamp": pd.Timestamp.now().isoformat(),
        "metadata": metadata or {},
    }
    if optimizer:
        save_data["optimizer_state_dict"] = optimizer.state_dict()
    if not filepath.endswith(".pt"):
        filepath += ".pt"
    torch.save(save_data, filepath)
    logger.info(f"Model saved to {filepath}")


def load_modern_model(filepath: str, model_class=None, **model_kwargs):
    if not filepath.endswith(".pt"):
        filepath += ".pt"
    checkpoint = torch.load(filepath, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if model_class:
            model = model_class(**model_kwargs)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError("model_class required for modern checkpoint format")
        logger.info(f"Loaded modern model from {filepath}")
        return model, checkpoint.get("metadata", {})
    else:
        logger.info(f"Loaded legacy model from {filepath}")
        return checkpoint, {}


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for handling chunked sequences with proper length control.
    Ensures sequences don't exceed model's maximum input length.
    """
    def __init__(self, sequences: List[List[int]], max_length: int = 1024):
        self.sequences = sequences
        self.max_length = max_length
        
        # Filter and truncate sequences to ensure they fit within max_length
        self.filtered_sequences = []
        for seq in sequences:
            if len(seq) > max_length:
                # Split long sequences into chunks
                for i in range(0, len(seq), max_length):
                    chunk = seq[i:i + max_length]
                    if len(chunk) > 1:  # Need at least 2 tokens (input + target)
                        self.filtered_sequences.append(chunk)
            elif len(seq) > 1:
                self.filtered_sequences.append(seq)
        
        logger.info(f"Dataset created with {len(self.filtered_sequences)} valid sequences")
        logger.info(f"Max sequence length: {max([len(seq) for seq in self.filtered_sequences])}")
        
    def __len__(self):
        return len(self.filtered_sequences)
    
    def __getitem__(self, idx):
        sequence = self.filtered_sequences[idx]
        # Create input (all tokens except last) and target (all tokens except first)
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        return input_seq, target_seq


def create_train_val_loaders(
    sequences: List[List[int]], 
    batch_size: int = 32, 
    max_length: int = 1024,
    val_split: float = 0.1,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders with proper sequence length handling.
    """
    # Split sequences into train and validation
    split_idx = int(len(sequences) * (1 - val_split))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    logger.info(f"Creating data loaders: {len(train_sequences)} train, {len(val_sequences)} val sequences")
    
    train_loader = create_data_loader(train_sequences, batch_size, max_length, shuffle=shuffle)
    val_loader = create_data_loader(val_sequences, batch_size, max_length, shuffle=False)
    
    return train_loader, val_loader


def create_data_loader(
    sequences: List[List[int]], 
    batch_size: int = 32, 
    max_length: int = 1024,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader with proper sequence length handling.
    """
    dataset = SequenceDataset(sequences, max_length=max_length)
    
    def collate_fn(batch):
        # Separate inputs and targets
        inputs, targets = zip(*batch)
        
        # Pad sequences to the same length within the batch
        max_len = min(max_length - 1, max([len(seq) for seq in inputs]))
        
        padded_inputs = []
        padded_targets = []
        
        for inp, tgt in zip(inputs, targets):
            # Truncate if necessary
            if len(inp) > max_len:
                inp = inp[:max_len]
                tgt = tgt[:max_len]
            
            # Pad if necessary  
            pad_length = max_len - len(inp)
            if pad_length > 0:
                inp = torch.cat([inp, torch.zeros(pad_length, dtype=torch.long)])
                tgt = torch.cat([tgt, torch.zeros(pad_length, dtype=torch.long)])
            
            padded_inputs.append(inp)
            padded_targets.append(tgt)
        
        return torch.stack(padded_inputs), torch.stack(padded_targets)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        drop_last=True  # Drop incomplete batches to maintain consistent shapes
    )


def analyze_dataset(data_path: Union[str, Path]) -> Dict:
    df = load_structured_data(data_path)
    analysis = {
        "total_dialogues": len(df),
        "unique_characters": df["Character"].nunique(),
        "character_counts": df["Character"].value_counts().to_dict(),
        "avg_dialogue_length": df["Dialogue"].str.len().mean(),
        "total_words": df["Dialogue"].str.split().str.len().sum(),
        "vocabulary_size": len(set(" ".join(df["Dialogue"]).lower().split())),
        "longest_dialogue": df["Dialogue"].str.len().max(),
        "shortest_dialogue": df["Dialogue"].str.len().min(),
    }
    logger.info("Dataset Analysis:")
    for key, value in analysis.items():
        if key != "character_counts":
            logger.info(f" {key}: {value}")
    return analysis


def create_character_embeddings(
    characters: List[str], embedding_dim: int = 64
) -> torch.Tensor:
    num_characters = len(characters)
    embeddings = torch.nn.Embedding(num_characters, embedding_dim)
    torch.nn.init.xavier_uniform_(embeddings.weight)
    logger.info(f"Created character embeddings: {num_characters} x {embedding_dim}")
    return embeddings # type: ignore
