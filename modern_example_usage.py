# modern_example_usage.py
# Example Usage: Modern TV Script Generation with Improved HelperAI
# Updated with sliding-window safety for generation to prevent token overflow

from typing import Optional
import torch
import torch.nn as nn

from improved_helperAI import (
    ModernTextProcessor,
    preprocess_and_save_modern_data,
    load_modern_preprocess,
    save_modern_model,
    load_modern_model,
    analyze_dataset,
    create_character_embeddings,
    SPECIAL_WORDS,
)


# ================================================================
# Example 1: Load and preprocess your dataset
# ================================================================


def example_modern_preprocessing():
    """
    Example showing modern preprocessing using advanced pipeline.
    """
    print("=== Modern Preprocessing Example ===")
    analysis = analyze_dataset("sample.xlsx")
    print("Dataset Stats:", analysis)

    preprocessed = preprocess_and_save_modern_data(
        data_path="sample.xlsx",
        output_path="modern_got_preprocess.pkl",
        tokenizer_type="gpt2",
        model_name="gpt2",
        min_frequency=1,
        max_vocab_size=5000,
        context_window=64,
    )
    print(f"Vocabulary size: {preprocessed['metadata']['vocab_size']}")
    print(f"Characters: {', '.join(preprocessed['metadata']['characters'])}")
    return preprocessed


# ================================================================
# Example 2: Modern Script Model with Attention
# ================================================================


class ModernScriptRNN(nn.Module):
    """
    Enhanced RNN with character embeddings + self-attention.
    """

    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        characters,
        dropout=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.characters = characters

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Character embeddings
        self.char_embedding = create_character_embeddings(
            characters, embedding_dim // 4
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim + embedding_dim // 4,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        # Output
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, input_seq, hidden, character_ids=None):
        batch_size, seq_len = input_seq.size()
        word_embeds = self.embedding(input_seq)

        if character_ids is not None:
            char_embeds = self.char_embedding(character_ids) # type: ignore
            char_embeds = char_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            embeds = torch.cat([word_embeds, char_embeds], dim=-1)
        else:
            char_padding = torch.zeros(
                batch_size,
                seq_len,
                word_embeds.size(-1) // 4,
                device=word_embeds.device,
            )
            embeds = torch.cat([word_embeds, char_padding], dim=-1)

        lstm_out, hidden = self.lstm(embeds, hidden)
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        lstm_out = self.norm(lstm_out + attended_out)
        lstm_out = self.dropout(lstm_out)

        output = self.fc1(lstm_out)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        final_output = output[:, -1, :]
        return final_output, hidden

    def init_hidden(self, batch_size, device="cpu"):
        return (
            torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(device),
        )


# ================================================================
# Example 3: Modern Trainer
# ================================================================


class ModernTrainer:
    def __init__(self, model, train_loader, vocab_to_int, int_to_vocab):
        self.model = model
        self.train_loader = train_loader
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            device = next(self.model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = self.model.init_hidden(inputs.size(0), device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs, _ = self.model(inputs, hidden)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, _ = self.model(inputs, hidden)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}")

        self.scheduler.step()
        return total_loss / len(self.train_loader)


# ================================================================
# Example 4: Modern Generator with Token Limit Fix
# ================================================================


class ModernGenerator:
    """
    Modern text generation with sliding-window trim to avoid length overflow.
    """

    def __init__(
        self, model, vocab_to_int, int_to_vocab, character_vocab=None, tokenizer=None
    ):
        self.model = model
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        self.character_vocab = character_vocab or {}
        self.tokenizer = tokenizer  # pass in tokenizer for encoding

    def generate_nucleus_sampling(
        self,
        seed_text: str,
        max_length: int = 200,
        top_p: float = 0.9,
        temperature: float = 1.0,
        character: Optional[str] = None,
    ) -> str:
        self.model.eval()
        device = next(self.model.parameters()).device

        # Determine safe window
        window = (
            getattr(self.tokenizer, "model_max_length", 1024)
            if self.tokenizer
            else 1024
        )
        if not window or window <= 0:
            window = 1024

        # Trim seed to max window
        if self.tokenizer:
            tokens = self.tokenizer.encode(seed_text, add_special_tokens=False)
        else:
            tokens = [
                self.vocab_to_int.get(tok, self.vocab_to_int[SPECIAL_WORDS["UNKNOWN"]])
                for tok in seed_text.split()
            ]
        sequence = tokens[-window:]

        generated_tokens = []
        for _ in range(max_length):
            input_ids = torch.tensor([sequence], device=device)
            hidden = self.model.init_hidden(1, device)

            char_id = None
            if character and character in self.character_vocab:
                idx = list(self.character_vocab.keys()).index(character)
                char_id = torch.tensor([idx], device=device)

            logits, _ = self.model(input_ids, hidden, char_id)
            logits = logits[0] / temperature

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")

            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated_tokens.append(next_token)
            sequence.append(next_token)
            if len(sequence) > window:
                sequence = sequence[-window:]

        words = [self.int_to_vocab.get(t, "<UNK>") for t in generated_tokens]
        text = " ".join(words)

        replace_map = {
            "||Period||": ".",
            "||Comma||": ",",
            "||Question||": "?",
            "||Exclamation||": "!",
            "||NewLine||": "\n",
            "||Quote||": '"',
            "||LParen||": "(",
            "||RParen||": ")",
        }
        for token, char in replace_map.items():
            text = text.replace(token.lower(), char)

        return text


# ================================================================
# Example 5: Complete workflow example
# ================================================================


def complete_modern_example():
    print("=== Complete Modern Example ===")
    pre = preprocess_and_save_modern_data(
        data_path="sample.xlsx",
        output_path="complete_preprocess.pkl",
        tokenizer_type="gpt2",
        model_name="gpt2",
        min_frequency=1,
        context_window=64,
    )
    vocab_size = pre["metadata"]["vocab_size"]
    chars = pre["metadata"]["characters"]

    model = ModernScriptRNN(
        vocab_size=vocab_size,
        output_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        n_layers=2,
        characters=chars,
        dropout=0.3,
    )

    save_modern_model("modern_script_model", model, metadata={"dataset": "Sample"})
    loaded_model, _ = load_modern_model(
        "modern_script_model",
        ModernScriptRNN,
        vocab_size=vocab_size,
        output_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        n_layers=2,
        characters=chars,
    )

    gen = ModernGenerator(
        model=loaded_model,
        vocab_to_int=pre["vocab_to_int"],
        int_to_vocab=pre["int_to_vocab"],
        character_vocab=pre["character_vocab"],
        tokenizer=ModernTextProcessor("gpt2", "gpt2").tokenizer,
    )

    for c in chars[:3]:
        print(f"\n--- {c.upper()} ---")
        print(
            gen.generate_nucleus_sampling(
                seed_text=f"{c}:", max_length=100, character=c
            )
        )


if __name__ == "__main__":
    try:
        # example_modern_preprocessing()
        complete_modern_example()
    except Exception as e:
        print("Error in example:", e)
