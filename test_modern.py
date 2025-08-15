# pytest-based unit tests for modern_plot.py and improved_helperAI.py

import os
import pickle
import pandas as pd
import pytest
import torch

from improved_helperAI import (
    load_structured_data,
    standardize_dataframe_columns,
    load_legacy_text_data,
    create_modern_lookup_tables,
    advanced_token_lookup,
    create_contextual_sequences,
    load_modern_preprocess,
    save_modern_model,
    load_modern_model,
    analyze_dataset,
    ModernTextProcessor,
)

from modern_plot import parse_log_file, plot_training_metrics

# ----------------------
# Fixtures
# ----------------------


@pytest.fixture
def sample_df(tmp_path):
    data = {"Name": ["a", "b"], "Sentence": ["Hello world.", "Testing unit tests!"]}
    file = tmp_path / "sample.csv"
    pd.DataFrame(data).to_csv(file, index=False)
    return file


@pytest.fixture
def sample_legacy_txt(tmp_path):
    text = "character1: Hello there\ncharacter2: General Kenobi!\nend"
    file = tmp_path / "legacy.txt"
    file.write_text(text)
    return file


@pytest.fixture
def sample_log(tmp_path):
    lines = [
        "Epoch: 1/5 Loss: 2.345 LR: 0.000300\n",
        "Epoch: 2/5 Loss: 2.123 LR: 0.000295\n",
        "Epoch: 3/5 Loss: 1.987 LR: 0.000290\n",
    ]
    file = tmp_path / "train.log"
    file.write_text("".join(lines))
    return file


# ----------------------
# improved_helperAI tests
# ----------------------


def test_load_structured_data(sample_df):
    df = load_structured_data(str(sample_df))
    assert "Character" in df.columns and "Dialogue" in df.columns
    assert len(df) == 2


def test_load_legacy_text_data(sample_legacy_txt):
    df = load_legacy_text_data(str(sample_legacy_txt))
    assert list(df["Character"]) == ["character1", "character2"]


def test_create_modern_lookup_tables():
    vocab_to_int, int_to_vocab = create_modern_lookup_tables(
        ["hello world", "hello again"], min_frequency=1, max_vocab_size=10
    )
    assert "hello" in vocab_to_int
    assert vocab_to_int[int_to_vocab[0]] == 0


def test_advanced_token_lookup():
    tokens = advanced_token_lookup()
    for symbol in [".", ",", "!", "?", "\n"]:
        assert symbol in tokens
        assert tokens[symbol].startswith("||")


def test_contextual_sequences():
    seq = list(range(10))
    window = 4
    sequences = create_contextual_sequences(seq, window)
    assert all(len(s) == window for s in sequences)


def test_analyze_dataset(sample_df):
    analysis = analyze_dataset(str(sample_df))
    assert "total_dialogues" in analysis
    assert analysis["total_dialogues"] == 2


# ----------------------
# Model save/load tests
# ----------------------


def test_model_save_load(tmp_path):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

    model = DummyModel()
    path = tmp_path / "dummy_model"
    save_modern_model(str(path), model)
    loaded_model, metadata = load_modern_model(str(path), DummyModel)
    assert isinstance(loaded_model, DummyModel)
    assert isinstance(metadata, dict)


# ----------------------
# ModernTextProcessor test
# ----------------------


def test_modern_text_processor_basic():
    processor = ModernTextProcessor(tokenizer_type="basic")
    assert processor.tokenizer is None
    processor = ModernTextProcessor(tokenizer_type="gpt2")
    if processor.tokenizer:  # Only if transformers available
        tokens = processor.tokenizer.tokenize("Hello world!")
        assert isinstance(tokens, list)


# ----------------------
# modern_plot tests
# ----------------------


def test_parse_log_file(sample_log):
    records = parse_log_file(str(sample_log))
    assert isinstance(records, list)
    for r in records:
        assert all(k in r for k in ("epoch", "batch", "loss", "lr"))


def test_plot_training_metrics_from_records(sample_log, tmp_path):
    records = parse_log_file(str(sample_log))
    out_path = tmp_path / "plot.png"
    plot_training_metrics(records, metrics=["loss", "lr"], save_path=str(out_path))
    assert out_path.exists()


def test_plot_training_metrics_from_file(sample_log, tmp_path):
    # save CSV from parsed records
    records = parse_log_file(str(sample_log))
    csv_path = tmp_path / "log.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    out_path = tmp_path / "plot.png"
    plot_training_metrics(
        str(csv_path), metrics=["loss", "lr"], save_path=str(out_path)
    )
    assert out_path.exists()


def test_plot_training_metrics_missing_col_error():
    bad_data = [{"epoch": 1, "loss": 0.5}]  # missing 'batch'
    with pytest.raises(ValueError):
        plot_training_metrics(bad_data, metrics=["loss"])
