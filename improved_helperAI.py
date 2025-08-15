# Modern TV Script Generation Helper Functions
# Implements advanced preprocessing, tokenization, and model utilities
# Updated for 2024 with Transformer support and modern NLP techniques

import os
import pickle
import json
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from collections import Counter

# Advanced tokenization and preprocessing
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BertTokenizer,
        BertModel,
        GPT2Tokenizer,
        GPT2LMHeadModel,
    )
    from datasets import Dataset

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Using basic tokenization.")

# Modern preprocessing libraries
try:
    import sentencepiece as spm

    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename="training.log",
    filemode="w",
    format="%(asctime)s %(levelname)s:%(message)s",
)
logger = logging.getLogger(__name__)

# Enhanced special tokens for modern preprocessing
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
    tokenizer_type: str = "bpe",
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
            tok: idx for idx, tok in processor.tokenizer.get_vocab().items()
        }
        int_to_vocab = {idx: tok for tok, idx in vocab_to_int.items()}

    # Enforce sliding-window max length
    sequences = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + context_window]
        if len(chunk) < context_window:
            pad_id = (
                processor.tokenizer.pad_token_id
                if processor.tokenizer
                else vocab_to_int[SPECIAL_WORDS["PADDING"]]
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
