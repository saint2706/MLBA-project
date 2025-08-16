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
    create_data_loader,
    create_train_val_loaders,
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
        context_window=64,  # Use smaller context window to prevent sequence length issues
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

        # Return full sequence outputs for training
        # For generation, we can get the last token with output[:, -1, :]
        return output, hidden

    def init_hidden(self, batch_size, device="cpu"):
        return (
            torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(device),
        )


# ================================================================
# Example 3: Modern Trainer
# ================================================================


class ModernTrainer:
    def __init__(self, model, train_loader, vocab_to_int, int_to_vocab, val_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        
        # Enhanced optimizer settings for longer training
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4,  # Lower initial learning rate for stability
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Better scheduler for long training
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=5e-4,
            epochs=200,  # Plan for many epochs
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)  # Ignore padding
        
        # Use the new PyTorch AMP API
        if torch.cuda.is_available():
            from torch.amp.grad_scaler import GradScaler
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # Training tracking
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 20  # Early stopping patience
        
        # Enhanced logging
        import logging
        import datetime
        
        # Configure detailed logging
        log_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.logger = logging.getLogger('ModernTrainer')
        self.logger.setLevel(logging.INFO)
        
        # File handler for training_output.txt
        file_handler = logging.FileHandler('training_output.txt', mode='w')
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("="*60)
        self.logger.info("MODERN TV SCRIPT GENERATOR TRAINING STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Vocabulary size: {len(vocab_to_int)}")
        self.logger.info(f"Training batches: {len(train_loader)}")
        self.logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        self.logger.info(f"Mixed Precision: {'Enabled' if self.scaler else 'Disabled'}")
        self.logger.info(f"Start time: {datetime.datetime.now()}")
        self.logger.info("="*60)

    def train_epoch(self, epoch):
        """Train for one epoch with comprehensive logging"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Log epoch start
        self.logger.info(f"\nEPOCH {epoch + 1}")
        self.logger.info("-" * 30)
        
        import time
        cpu_start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            device = next(self.model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = self.model.init_hidden(inputs.size(0), device)

            if self.scaler:
                from torch.amp.autocast_mode import autocast
                with autocast('cuda'):
                    outputs, _ = self.model(inputs, hidden)
                    
                    # Debug tensor shapes
                    if batch_idx == 0 and epoch == 0:
                        self.logger.info(f"DEBUG - Input shape: {inputs.shape}")
                        self.logger.info(f"DEBUG - Targets shape: {targets.shape}")
                        self.logger.info(f"DEBUG - Outputs shape: {outputs.shape}")
                    
                    # Ensure shapes match for proper flattening
                    if outputs.dim() == 3 and targets.dim() == 2:
                        # outputs: [batch, seq_len, vocab_size], targets: [batch, seq_len]
                        outputs_flat = outputs.view(-1, outputs.size(-1))  # [batch*seq_len, vocab_size]
                        targets_flat = targets.view(-1)  # [batch*seq_len]
                    else:
                        # Handle mismatched dimensions
                        self.logger.error(f"Shape mismatch: outputs {outputs.shape}, targets {targets.shape}")
                        raise ValueError(f"Unexpected tensor shapes: outputs {outputs.shape}, targets {targets.shape}")
                    
                    loss = self.criterion(outputs_flat, targets_flat)
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, _ = self.model(inputs, hidden)
                
                # Debug tensor shapes
                if batch_idx == 0 and epoch == 0:
                    self.logger.info(f"DEBUG - Input shape: {inputs.shape}")
                    self.logger.info(f"DEBUG - Targets shape: {targets.shape}")
                    self.logger.info(f"DEBUG - Outputs shape: {outputs.shape}")
                
                # Ensure shapes match for proper flattening
                if outputs.dim() == 3 and targets.dim() == 2:
                    # outputs: [batch, seq_len, vocab_size], targets: [batch, seq_len]
                    outputs_flat = outputs.view(-1, outputs.size(-1))  # [batch*seq_len, vocab_size]
                    targets_flat = targets.view(-1)  # [batch*seq_len]
                else:
                    # Handle mismatched dimensions
                    self.logger.error(f"Shape mismatch: outputs {outputs.shape}, targets {targets.shape}")
                    raise ValueError(f"Unexpected tensor shapes: outputs {outputs.shape}, targets {targets.shape}")
                
                loss = self.criterion(outputs_flat, targets_flat)
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()  # Step per batch for OneCycleLR
            
            total_loss += loss.item()
            
            # Detailed logging every 10 batches
            if batch_idx % 10 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss_so_far = total_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / num_batches * 100
                
                self.logger.info(
                    f"Batch {batch_idx:4d}/{num_batches} ({progress:5.1f}%) | "
                    f"Loss: {loss.item():.6f} | Avg Loss: {avg_loss_so_far:.6f} | "
                    f"LR: {current_lr:.2e} | Grad Norm: {grad_norm:.4f}"
                )
            
            # Memory management
            if torch.cuda.is_available():
                if batch_idx % 50 == 0:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    self.logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        # Calculate epoch timing
        cpu_end_time = time.time()
        cpu_epoch_time = cpu_end_time - cpu_start_time
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Log epoch summary
        self.logger.info(f"\nEPOCH {epoch + 1} SUMMARY:")
        self.logger.info(f"Average Loss: {avg_loss:.6f}")
        self.logger.info(f"Total Batches: {num_batches}")
        self.logger.info(f"Epoch Time: {cpu_epoch_time:.2f}s")
        self.logger.info(f"Current Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate the model and log results"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_val_loss = 0.0
        
        self.logger.info(f"\nValidation for Epoch {epoch + 1}")
        self.logger.info("-" * 25)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                device = next(self.model.parameters()).device
                inputs, targets = inputs.to(device), targets.to(device)
                hidden = self.model.init_hidden(inputs.size(0), device)
                
                outputs, _ = self.model(inputs, hidden)
                
                # Debug tensor shapes for validation
                if batch_idx == 0 and epoch == 0:
                    self.logger.info(f"VAL DEBUG - Input shape: {inputs.shape}")
                    self.logger.info(f"VAL DEBUG - Targets shape: {targets.shape}")
                    self.logger.info(f"VAL DEBUG - Outputs shape: {outputs.shape}")
                
                # Ensure shapes match for proper flattening
                if outputs.dim() == 3 and targets.dim() == 2:
                    outputs_flat = outputs.view(-1, outputs.size(-1))  # [batch*seq_len, vocab_size]
                    targets_flat = targets.view(-1)  # [batch*seq_len]
                else:
                    # Handle mismatched dimensions
                    self.logger.error(f"VAL Shape mismatch: outputs {outputs.shape}, targets {targets.shape}")
                    raise ValueError(f"VAL Unexpected tensor shapes: outputs {outputs.shape}, targets {targets.shape}")
                
                loss = self.criterion(outputs_flat, targets_flat)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        self.logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'vocab_to_int': self.vocab_to_int,
            'int_to_vocab': self.int_to_vocab
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, 'checkpoint_latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, 'checkpoint_best.pt')
            self.logger.info(f"✓ New best model saved! (Loss: {loss:.6f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
            self.logger.info(f"✓ Checkpoint saved for epoch {epoch+1}")
    
    def generate_sample_text(self, generator, epoch, max_samples=3):
        """Generate sample text during training to monitor progress"""
        self.logger.info(f"\nSample Generation (Epoch {epoch + 1}):")
        self.logger.info("-" * 40)
        
        sample_prompts = ["TYRION:", "JON:", "DAENERYS:"]
        
        self.model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(sample_prompts[:max_samples]):
                try:
                    sample = generator.generate_nucleus_sampling(
                        seed_text=prompt,
                        max_length=50,
                        temperature=0.8,
                        top_p=0.9
                    )
                    self.logger.info(f"Sample {i+1}: {prompt} {sample}")
                except Exception as e:
                    self.logger.warning(f"Sample generation failed: {e}")
        
        self.model.train()
    
    def train_full(self, num_epochs, generator=None, save_every=10, sample_every=5):
        """Complete training loop with all enhancements"""
        self.logger.info(f"\nStarting full training for {num_epochs} epochs")
        self.logger.info(f"Save checkpoints every {save_every} epochs")
        self.logger.info(f"Generate samples every {sample_every} epochs")
        self.logger.info("="*60)
        
        import time
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            
            # Check for improvement
            current_loss = val_loss if val_loss is not None else train_loss
            is_best = current_loss < self.best_loss
            
            if is_best:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoints
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, current_loss, is_best)
            
            # Generate samples
            if generator and (epoch + 1) % sample_every == 0:
                self.generate_sample_text(generator, epoch)
            
            # Log progress
            elapsed_time = time.time() - training_start_time
            estimated_total = (elapsed_time / (epoch + 1)) * num_epochs
            remaining_time = estimated_total - elapsed_time
            
            self.logger.info(f"\nProgress: {epoch + 1}/{num_epochs} ({(epoch + 1)/num_epochs*100:.1f}%)")
            self.logger.info(f"Best Loss: {self.best_loss:.6f} | Current Loss: {current_loss:.6f}")
            self.logger.info(f"Time Elapsed: {elapsed_time/3600:.2f}h | ETA: {remaining_time/3600:.2f}h")
            self.logger.info(f"Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                self.logger.info(f"No improvement for {self.patience} epochs")
                break
        
        total_training_time = time.time() - training_start_time
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING COMPLETED!")
        self.logger.info("="*60)
        self.logger.info(f"Total training time: {total_training_time/3600:.2f} hours")
        self.logger.info(f"Final best loss: {self.best_loss:.6f}")
        self.logger.info(f"Total epochs: {epoch + 1}/{num_epochs}")
        
        return self.best_loss


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
    
    def decode_custom_tokens(self, text: str) -> str:
        """Convert custom tokens back to punctuation"""
        token_map = {
            "||period||": ".",
            "||comma||": ",", 
            "||exclamation||": "!",
            "||question||": "?",
            "||semicolon||": ";",
            "||colon||": ":",
            "||quote||": '"',
            "||apostrophe||": "'",
            "||lparen||": "(",
            "||rparen||": ")",
            "||lbracket||": "[",
            "||rbracket||": "]",
            "||dash||": "-",
            "||ndash||": "–",
            "||mdash||": "—",
            "||newline||": "\n",
            "||tab||": "\t",
            "||ellipsis||": "...",
            "||slash||": "/",
            "||ampersand||": "&",
            "||at||": "@",
            "||hash||": "#",
            "||dollar||": "$",
            "||percent||": "%",
            "||star||": "*",
            "||plus||": "+",
            "||equal||": "=",
            "||less||": "<",
            "||greater||": ">",
            "||caret||": "^",
            "||underscore||": "_",
            "||pipe||": "|",
            "||tilde||": "~",
            "||backtick||": "`",
        }
        
        for token, symbol in token_map.items():
            text = text.replace(token, symbol)
        
        return text

    def generate_nucleus_sampling(
        self,
        seed_text: str,
        max_length: int = 200,
        top_p: float = 0.9,
        temperature: float = 1.0,
        character: Optional[str] = None,
        repetition_penalty: float = 1.2,
    ) -> str:
        self.model.eval()
        device = next(self.model.parameters()).device

        # Set a safe maximum sequence length - use a conservative limit
        MAX_SEQUENCE_LENGTH = 64  # Match the context window used in preprocessing
        
        # Encode and trim seed to maximum allowed window size
        if self.tokenizer:
            tokens = self.tokenizer.encode(seed_text, add_special_tokens=False)
        else:
            tokens = [
                self.vocab_to_int.get(tok, self.vocab_to_int[SPECIAL_WORDS["UNKNOWN"]])
                for tok in seed_text.split()
            ]
        
        # Ensure we don't exceed the maximum sequence length
        sequence = tokens[-MAX_SEQUENCE_LENGTH:]

        generated_tokens = []
        for _ in range(max_length):
            # Ensure sequence doesn't exceed max length
            if len(sequence) > MAX_SEQUENCE_LENGTH:
                sequence = sequence[-MAX_SEQUENCE_LENGTH:]
                
            input_ids = torch.tensor([sequence], device=device)
            hidden = self.model.init_hidden(1, device)

            char_id = None
            if character and character in self.character_vocab:
                idx = list(self.character_vocab.keys()).index(character)
                char_id = torch.tensor([idx], device=device)

            logits, _ = self.model(input_ids, hidden, char_id)
            # Get the last token's logits for next token prediction
            logits = logits[0, -1, :] / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_tokens:
                for token_id in set(generated_tokens[-50:]):  # Only consider recent tokens
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

            # Nucleus (top-p) sampling filtering
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
            # Keep sequence within bounds
            if len(sequence) > MAX_SEQUENCE_LENGTH:
                sequence = sequence[-MAX_SEQUENCE_LENGTH:]

        # Decode generated tokens to human-readable text using the tokenizer
        if self.tokenizer is None:
            # fallback for basic tokenizer - convert tokens to words using int_to_vocab
            words = [self.int_to_vocab.get(t, "<UNK>") for t in generated_tokens]
            text = " ".join(words)
        else:
            # Use proper decoding for transformers tokenizer
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)  # type: ignore

        return text


# ================================================================
# Example 5: Complete workflow example
# ================================================================


def complete_modern_example():
    print("=== Complete Modern Example with Extended Training ===")
    pre = preprocess_and_save_modern_data(
        data_path="sample.xlsx",
        output_path="complete_preprocess.pkl",
        tokenizer_type="gpt2",
        model_name="gpt2",
        min_frequency=1,
        context_window=64,  # Use smaller context window to avoid sequence length issues
    )
    vocab_size = pre["metadata"]["vocab_size"]
    chars = pre["metadata"]["characters"]

    # Create train/validation DataLoaders with proper sequence length handling
    train_loader, val_loader = create_train_val_loaders(
        sequences=pre["sequences"],
        batch_size=16,  # Smaller batch size for stability
        max_length=64,  # Match the context window
        val_split=0.1,  # 10% validation split
        shuffle=True
    )

    # Create model with CPU/GPU detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = ModernScriptRNN(
        vocab_size=vocab_size,
        output_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        n_layers=2,
        characters=chars,
        dropout=0.3,
    ).to(device)

    # Create trainer with validation support
    trainer = ModernTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # Add validation loader
        vocab_to_int=pre["vocab_to_int"],
        int_to_vocab=pre["int_to_vocab"]
    )
    
    # Create generator for sampling during training
    gen = ModernGenerator(
        model=model,
        vocab_to_int=pre["vocab_to_int"],
        int_to_vocab=pre["int_to_vocab"],
        character_vocab=pre["character_vocab"],
        tokenizer=ModernTextProcessor("gpt2", "gpt2").tokenizer,
    )
    
    print("Starting extended training session...")
    print("Training will run for up to 200 epochs (can take 8+ hours)")
    print("Check training_output.txt for detailed progress logs")
    print("Checkpoints will be saved every 10 epochs")
    print("Sample text will be generated every 5 epochs")
    
    # Train for many epochs with comprehensive logging
    final_loss = trainer.train_full(
        num_epochs=200,  # Extended training for quality output
        generator=gen,   # Generate samples during training
        save_every=10,   # Save checkpoints every 10 epochs
        sample_every=5   # Generate samples every 5 epochs
    )

    print(f"\nTraining completed! Final best loss: {final_loss:.6f}")
    
    # Load the best model for final evaluation
    try:
        checkpoint = torch.load('checkpoint_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded best model from checkpoint")
    except FileNotFoundError:
        print("⚠️  Best checkpoint not found, using current model")

    # Save the final model
    save_modern_model("modern_script_model", model, metadata={"dataset": "Sample", "final_loss": final_loss})
    
    # Generate final samples with different characters and settings
    print("\n" + "="*60)
    print("FINAL SAMPLE GENERATION")
    print("="*60)
    
    model.eval()
    for i, c in enumerate(chars[:5]):  # Test more characters
        print(f"\n--- {c.upper()} (Temperature 0.7) ---")
        sample1 = gen.generate_nucleus_sampling(
            seed_text=f"{c}:",
            max_length=150,
            temperature=0.7,
            top_p=0.9,
            character=c
        )
        print(sample1)
        
        print(f"\n--- {c.upper()} (Temperature 1.0) ---")
        sample2 = gen.generate_nucleus_sampling(
            seed_text=f"{c}:",
            max_length=100,
            temperature=1.0,
            top_p=0.8,
            character=c
        )
        print(sample2)


if __name__ == "__main__":
    try:
        # example_modern_preprocessing()
        complete_modern_example()
    except Exception as e:
        print("Error in example:", e)
