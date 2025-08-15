# main_modern.py
# Modernized main script for TV script generation
# Uses improved_helperAI and modern architecture with context length safety

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from improved_helperAI import (
    preprocess_and_save_modern_data,
    load_modern_preprocess,
    save_modern_model,
    load_modern_model,
    analyze_dataset,
)

from modern_example_usage import (
    ModernScriptRNN,
    ModernTrainer,
    ModernGenerator,
    ModernTextProcessor,
)

# ================================================================
# 1. Configuration
# ================================================================
DATA_PATH = "data/Game_of_Thrones_Script.csv"  # <-- change to your dataset
PREPROCESS_PATH = "preprocess_modern.pkl"
MODEL_SAVE_PATH = "modern_script_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER_TYPE = "gpt2"
MODEL_NAME = "gpt2"
MIN_FREQUENCY = 2
MAX_VOCAB_SIZE = 10000
CONTEXT_WINDOW = 1024  # safe max length for GPT-2
BATCH_SIZE = 16  # reduce if you hit GPU RAM issues
NUM_EPOCHS = 5
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 3e-4
TOP_P = 0.9
TEMPERATURE = 0.8
GEN_LENGTH = 200

# ================================================================
# 2. Data Analysis
# ================================================================
analyze_dataset(DATA_PATH)

# ================================================================
# 3. Preprocessing
# ================================================================
preprocessed = preprocess_and_save_modern_data(
    data_path=DATA_PATH,
    output_path=PREPROCESS_PATH,
    tokenizer_type=TOKENIZER_TYPE,
    model_name=MODEL_NAME,
    min_frequency=MIN_FREQUENCY,
    max_vocab_size=MAX_VOCAB_SIZE,
    context_window=CONTEXT_WINDOW,
)

# ================================================================
# 4. Prepare DataLoader
# ================================================================
sequences = np.array(
    preprocessed["sequences"]
)  # already trimmed/padded in preprocessing
# Prepare input X and target y (next-token prediction)
data_x = torch.tensor(sequences[:, :-1], dtype=torch.long)
data_y = torch.tensor(sequences[:, -1], dtype=torch.long)
tensor_data = TensorDataset(data_x, data_y)

train_loader = DataLoader(
    tensor_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

# ================================================================
# 5. Model Initialization
# ================================================================
vocab_size = preprocessed["metadata"]["vocab_size"]
characters = preprocessed["metadata"]["characters"]

model = ModernScriptRNN(
    vocab_size=vocab_size,
    output_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    n_layers=NUM_LAYERS,
    characters=characters,
    dropout=DROPOUT,
).to(DEVICE)

# ================================================================
# 6. Training Setup
# ================================================================
trainer = ModernTrainer(
    model, train_loader, preprocessed["vocab_to_int"], preprocessed["int_to_vocab"]
)

# ================================================================
# 7. Train Loop
# ================================================================
for epoch in range(1, NUM_EPOCHS + 1):
    avg_loss = trainer.train_epoch(epoch)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

# ================================================================
# 8. Save Model
# ================================================================
metadata = {
    "dataset": os.path.basename(DATA_PATH),
    "tokenizer": TOKENIZER_TYPE,
    "context_window": CONTEXT_WINDOW,
}
save_modern_model(
    MODEL_SAVE_PATH, model, optimizer=trainer.optimizer, metadata=metadata
)

# ================================================================
# 9. Load Model for Generation
# ================================================================
loaded_model, load_meta = load_modern_model(
    MODEL_SAVE_PATH,
    ModernScriptRNN,
    vocab_size=vocab_size,
    output_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    n_layers=NUM_LAYERS,
    characters=characters,
)
loaded_model.to(DEVICE) # type: ignore

# ================================================================
# 10. Generation
# ================================================================
generator = ModernGenerator(
    model=loaded_model,
    vocab_to_int=preprocessed["vocab_to_int"],
    int_to_vocab=preprocessed["int_to_vocab"],
    character_vocab=preprocessed["character_vocab"],
    tokenizer=ModernTextProcessor(TOKENIZER_TYPE, MODEL_NAME).tokenizer,
)

for char in characters[:4]:
    print(f"\n--- Generated for {char.upper()} ---")
    script = generator.generate_nucleus_sampling(
        seed_text=f"{char}:",
        max_length=GEN_LENGTH,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        character=char,
    )
    print(script)

print("\n✅ Script generation complete.")
