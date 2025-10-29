# main_modern.py
# 🎬 MAIN SCRIPT FOR TV SCRIPT GENERATION
# 
# FOR NON-PROGRAMMERS:
# This is the main control center for our Game of Thrones script generator.
# Think of it like the "master control panel" that coordinates everything:
# 1. Loads and prepares the Game of Thrones dialogue data
# 2. Trains an AI model to learn the writing patterns
# 3. Generates new script dialogue in the style of the show
#
# The AI learns by reading thousands of lines of real Game of Thrones dialogue
# and then tries to write new lines that sound like they could be from the show.
#
# Modernized main script for TV script generation
# Uses improved_helperAI and modern architecture with context length safety

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import datetime

# Import our custom helper functions (like having specialized tools)
from .improved_helperAI import (
    preprocess_and_save_modern_data,  # Prepares the raw text data for AI training
    load_modern_preprocess,           # Loads previously prepared data
    save_modern_model,               # Saves our trained AI model
    load_modern_model,               # Loads a previously trained AI model
    analyze_dataset,                 # Analyzes the Game of Thrones data
)

# Import our AI model classes (the actual "brain" of our system)
from .modern_example_usage import (
    ModernScriptRNN,      # The neural network that learns to write scripts
    ModernTrainer,        # The system that teaches the AI
    ModernGenerator,      # The system that generates new script text
    ModernTextProcessor,  # Handles text processing and tokenization
)

# ================================================================
# 🎛️ CONFIGURATION SETTINGS (The Control Panel)
# FOR NON-PROGRAMMERS: These are the "settings" for our AI system
# ================================================================

# 📁 File paths - where to find our data and save results
DATA_PATH = "data/Game_of_Thrones_Script.csv"  # ← Your Game of Thrones dialogue data
PREPROCESS_PATH = "preprocess_modern.pkl"     # Where preprocessed data is saved
MODEL_SAVE_PATH = "modern_script_model"       # Where the trained AI model is saved

# 🖥️ Computing setup - CPU or GPU (graphics card)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ================================================================
# 📝 LOGGING SETUP (Track progress and save to file)
# FOR NON-PROGRAMMERS: This creates detailed logs of everything that happens
# ================================================================

def setup_logging():
    """Set up comprehensive logging to training_output.txt"""
    # Create a logger
    logger = logging.getLogger('GameOfThronesAI')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters for different outputs
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter('%(message)s')
    
    # File handler - saves detailed logs to training_output.txt
    file_handler = logging.FileHandler('training_output.txt', mode='w', encoding='utf-8')
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler - shows simple messages on screen
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logging
logger = setup_logging()
logger.info("🎬 Game of Thrones AI Script Generator Started")
logger.info(f"🖥️  Device: {DEVICE}")

# 📝 Text processing settings
TOKENIZER_TYPE = "custom"      # Use custom vocabulary instead of GPT-2 (FIXED: vocabulary mismatch)
MODEL_NAME = "custom"          # Custom model for Game of Thrones data
MIN_FREQUENCY = 3              # Higher threshold for cleaner vocabulary (was 2)
MAX_VOCAB_SIZE = 15000         # Reasonable vocabulary size (was 10000, GPT-2 was 50257)
CONTEXT_WINDOW = 128           # Larger context for better dialogue coherence (was 64)

# 🎓 Training settings - how the AI learns
BATCH_SIZE = 8                 # Smaller batches to accommodate larger sequences (was 16)
NUM_EPOCHS = 200               # How many times to go through all the data (INCREASED for better quality)
EMBEDDING_DIM = 384            # Larger word representations for better quality (was 256)
HIDDEN_DIM = 768               # Larger AI "memory" for better context (was 512)
NUM_LAYERS = 3                 # More processing layers (was 2)
DROPOUT = 0.2                  # Reduced dropout for better retention (was 0.3)
LR = 1e-4                      # Learning rate - how fast the AI adapts (REDUCED)

# 🎭 Generation settings - how the AI creates new text
TOP_P = 0.95                   # Less restrictive nucleus sampling for more variety (was 0.9)
TEMPERATURE = 0.7              # Lower temperature for more coherent output (was 0.8)
GEN_LENGTH = 200               # Length of generated text
REPETITION_PENALTY = 1.2       # NEW: Penalize repeated tokens to reduce loops

# Log configuration after variables are defined
logger.info(f"⚙️  Configuration: {BATCH_SIZE} batch size, {NUM_EPOCHS} epochs, {CONTEXT_WINDOW} context window")

print("⚙️ Configuration loaded successfully!")
print(f"   📊 Batch size: {BATCH_SIZE}")
print(f"   🎓 Training epochs: {NUM_EPOCHS}")
print(f"   📏 Context window: {CONTEXT_WINDOW} tokens")
print(f"   🧠 Model dimensions: {EMBEDDING_DIM}d embeddings, {HIDDEN_DIM}d hidden")

# ================================================================
# 📊 STEP 1: DATA ANALYSIS (Understanding our Game of Thrones data)
# FOR NON-PROGRAMMERS: Before training, let's see what we're working with
# ================================================================

def analyze_got_data():
    """
    🔍 Analyze the Game of Thrones dataset
    
    This function examines our training data to understand:
    - How many characters speak in the show
    - How much dialogue each character has
    - The vocabulary size and complexity
    - Other important statistics
    """
    print("\n🔍 ANALYZING GAME OF THRONES DATA")
    print("=" * 50)
    
    try:
        analysis = analyze_dataset(DATA_PATH)
        
        print("📈 DATASET STATISTICS:")
        print(f"   💬 Total dialogues: {analysis.get('total_dialogues', 'Unknown')}")
        print(f"   👥 Unique characters: {analysis.get('unique_characters', 'Unknown')}")
        print(f"   📚 Vocabulary size: {analysis.get('vocabulary_size', 'Unknown')}")
        print(f"   📏 Average dialogue length: {analysis.get('avg_dialogue_length', 0):.1f} characters")
        print(f"   📝 Total words: {analysis.get('total_words', 'Unknown')}")
        
        # Show top characters
        char_counts = analysis.get('character_counts', {})
        if char_counts:
            print("\n🏆 TOP 10 CHARACTERS BY DIALOGUE:")
            for i, (char, count) in enumerate(list(char_counts.items())[:10], 1):
                print(f"   {i:2d}. {char}: {count} lines")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        print("💡 Make sure your data file exists at:", DATA_PATH)
        return None

# ================================================================
# 📝 STEP 2: DATA PREPROCESSING (Preparing text for AI training)
# FOR NON-PROGRAMMERS: This converts human text into numbers the AI can understand
# ================================================================

def preprocess_got_data():
    """
    📝 Preprocess Game of Thrones data for AI training
    
    This step converts the human-readable text into numbers and sequences
    that our AI model can learn from. Think of it like translating text
    into a language the computer can understand.
    """
    print("\n📝 PREPROCESSING GAME OF THRONES DATA")
    print("=" * 50)
    print("🔄 Converting text to AI-readable format...")
    print("⏳ This may take a few minutes for large datasets...")
    
    try:
        preprocessed = preprocess_and_save_modern_data(
            data_path=DATA_PATH,
            output_path=PREPROCESS_PATH,
            tokenizer_type=TOKENIZER_TYPE,
            model_name=MODEL_NAME,
            min_frequency=MIN_FREQUENCY,
            max_vocab_size=MAX_VOCAB_SIZE,
            context_window=CONTEXT_WINDOW,
        )
        
        # Display preprocessing results
        metadata = preprocessed.get('metadata', {})
        print("✅ PREPROCESSING COMPLETE!")
        print(f"   🎯 Vocabulary size: {metadata.get('vocab_size', 'Unknown')}")
        print(f"   🔤 Context window: {metadata.get('window', 'Unknown')} tokens")
        print(f"   📦 Number of sequences: {metadata.get('num_sequences', 'Unknown')}")
        print(f"   👥 Characters found: {len(metadata.get('characters', []))}")
        
        # Show character list
        characters = metadata.get('characters', [])
        if characters:
            print(f"   🎭 Main characters: {', '.join(characters[:10])}")
            if len(characters) > 10:
                print(f"      ... and {len(characters) - 10} more")
        
        print(f"💾 Preprocessed data saved to: {PREPROCESS_PATH}")
        return preprocessed
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        print("💡 Check your data file format and try again")
        return None

# ================================================================
# 🔧 STEP 3: PREPARE DATA FOR TRAINING (Creating AI-friendly data batches)
# FOR NON-PROGRAMMERS: This organizes our data into "batches" for efficient AI training
# ================================================================

def prepare_training_data(preprocessed_data):
    """
    🔧 Prepare training data batches
    
    This function takes our preprocessed data and organizes it into batches
    that our AI can efficiently learn from. Think of it like organizing 
    study materials into manageable chunks.
    """
    print("\n🔧 PREPARING TRAINING DATA")
    print("=" * 50)
    
    try:
        # Extract sequences from preprocessed data
        sequences = np.array(preprocessed_data["sequences"])
        print(f"📦 Processing {len(sequences)} training sequences...")
        
        # Prepare input-output pairs for training
        # Input: all tokens except the last one
        # Target: all tokens except the first one (what we want to predict)
        data_x = torch.tensor(sequences[:, :-1], dtype=torch.long)
        data_y = torch.tensor(sequences[:, 1:], dtype=torch.long)  # Fixed: should be next tokens
        
        print(f"   📥 Input shape: {data_x.shape}")
        print(f"   📤 Target shape: {data_y.shape}")
        
        # Create dataset and dataloader
        tensor_data = TensorDataset(data_x, data_y)
        
        # Set num_workers based on platform (Windows requires 0 to avoid multiprocessing issues)
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        
        train_loader = DataLoader(
            tensor_data, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            pin_memory=True if DEVICE.type == 'cuda' else False,
            num_workers=num_workers  # Platform-safe parallel data loading
        )
        
        print(f"✅ Created {len(train_loader)} training batches")
        print(f"   🎯 Batch size: {BATCH_SIZE}")
        return train_loader
        
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
        return None

# ================================================================
# 🧠 STEP 4: MODEL INITIALIZATION (Creating the AI brain)
# FOR NON-PROGRAMMERS: This creates the AI model that will learn to write scripts
# ================================================================

def create_ai_model(preprocessed_data):
    """
    🧠 Create and initialize the AI model
    
    This function creates our neural network - the "brain" that will learn
    to generate Game of Thrones dialogue. It's like building a complex
    pattern-recognition system that understands language.
    """
    print("\n🧠 CREATING AI MODEL")
    print("=" * 50)
    
    try:
        # Extract model parameters from preprocessed data
        metadata = preprocessed_data["metadata"]
        vocab_size = metadata["vocab_size"]
        characters = metadata["characters"]
        
        print(f"🎯 Model Configuration:")
        print(f"   📚 Vocabulary size: {vocab_size:,} words")
        print(f"   👥 Number of characters: {len(characters)}")
        print(f"   🧠 Embedding dimensions: {EMBEDDING_DIM}")
        print(f"   💭 Hidden dimensions: {HIDDEN_DIM}")
        print(f"   📚 Number of layers: {NUM_LAYERS}")
        
        # Create the neural network model
        model = ModernScriptRNN(
            vocab_size=vocab_size,
            output_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            n_layers=NUM_LAYERS,
            characters=characters,
            dropout=DROPOUT
        )
        
        # Move model to appropriate device (CPU or GPU)
        model = model.to(DEVICE)
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created successfully!")
        print(f"   🔢 Total parameters: {total_params:,}")
        print(f"   🎓 Trainable parameters: {trainable_params:,}")
        print(f"   💾 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"   🖥️ Device: {DEVICE}")
        
        return model, vocab_size, characters
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None, None, None

# ================================================================
# 🎓 STEP 5: TRAINING SETUP (Preparing the AI teacher)
# FOR NON-PROGRAMMERS: This creates the system that will teach our AI how to write
# ================================================================

def setup_trainer(model, train_loader, preprocessed_data):
    """
    🎓 Set up the AI training system
    
    This creates the "teacher" system that will guide our AI through
    the learning process. It handles the actual teaching and keeps
    track of how well the AI is learning.
    """
    print("\n🎓 SETTING UP AI TRAINER")
    print("=" * 50)
    
    try:
        trainer = ModernTrainer(
            model=model,
            train_loader=train_loader, 
            vocab_to_int=preprocessed_data["vocab_to_int"], 
            int_to_vocab=preprocessed_data["int_to_vocab"]
        )
        
        print("✅ Trainer initialized successfully!")
        print(f"   🎯 Learning rate: {LR}")
        print(f"   📊 Batch size: {BATCH_SIZE}")
        print(f"   🔄 Training epochs planned: {NUM_EPOCHS}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Error setting up trainer: {e}")
        return None

# ================================================================
# 🚀 STEP 6: MAIN TRAINING LOOP (Teaching the AI)
# FOR NON-PROGRAMMERS: This is where the actual learning happens!
# ================================================================

def train_ai_model(trainer, num_epochs, model, preprocessed_data, vocab_size, characters):
    """
    🚀 Train the AI model
    
    This is the main learning process where our AI studies thousands
    of Game of Thrones dialogue lines and learns the patterns.
    It's like a very intensive writing class for the AI!
    """
    print("\n🚀 STARTING AI TRAINING")
    print("=" * 50)
    print("📚 The AI will now learn from Game of Thrones dialogue...")
    print("⏰ This may take several hours - perfect time for a coffee break!")
    print(f"🎯 Target: {num_epochs} epochs")
    print(f"📝 Check training_output.txt for detailed progress logs")
    print()
    
    # Log training start
    logger.info("🚀 TRAINING STARTED")
    logger.info(f"📊 Target epochs: {num_epochs}")
    logger.info(f"📊 Batch size: {BATCH_SIZE}")
    logger.info(f"📊 Learning rate: {LR}")
    start_time = datetime.datetime.now()
    logger.info(f"⏰ Training start time: {start_time}")
    
    training_losses = []
    best_loss = float('inf')
    
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start = datetime.datetime.now()
            print(f"📖 Epoch {epoch}/{num_epochs} - Teaching the AI...")
            logger.info(f"📖 EPOCH {epoch}/{num_epochs} STARTED")
            
            # Train for one epoch
            avg_loss = trainer.train_epoch(epoch - 1)  # trainer expects 0-based epochs
            training_losses.append(avg_loss)
            
            epoch_end = datetime.datetime.now()
            epoch_time = (epoch_end - epoch_start).total_seconds()
            
            # Detailed logging for each epoch
            logger.info(f"📊 Epoch {epoch} completed in {epoch_time:.1f}s")
            logger.info(f"📊 Average loss: {avg_loss:.6f}")
            
            # Check if this is our best performance so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                improvement = (best_loss - avg_loss) if best_loss != float('inf') else 0
                print(f"   🏆 New best performance! Loss: {avg_loss:.6f}")
                logger.info(f"🏆 NEW BEST LOSS: {avg_loss:.6f} (improvement: {improvement:.6f})")
                
                # Save the best model
                save_modern_model(
                    f"{MODEL_SAVE_PATH}_best",
                    model, 
                    metadata={
                        "epoch": epoch,
                        "loss": avg_loss,
                        "vocab_size": vocab_size,
                        "characters": len(characters) if characters else 0
                    }
                )
                logger.info(f"💾 Best model saved to {MODEL_SAVE_PATH}_best")
            else:
                print(f"   📊 Current loss: {avg_loss:.6f} (Best: {best_loss:.6f})")
                logger.info(f"📊 Loss: {avg_loss:.6f} (Best remains: {best_loss:.6f})")
            
            # Calculate progress and ETA
            progress_pct = (epoch / num_epochs) * 100
            elapsed_time = (epoch_end - start_time).total_seconds()
            if epoch > 0:
                time_per_epoch = elapsed_time / epoch
                remaining_epochs = num_epochs - epoch
                eta_seconds = remaining_epochs * time_per_epoch
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(f"📈 Progress: {progress_pct:.1f}% | ETA: {eta_str}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                save_modern_model(
                    f"{MODEL_SAVE_PATH}_epoch_{epoch}",
                    model,
                    metadata={"epoch": epoch, "loss": avg_loss}
                )
                print(f"   💾 Checkpoint saved at epoch {epoch}")
                logger.info(f"💾 Checkpoint saved: {MODEL_SAVE_PATH}_epoch_{epoch}")
            
            # Generate sample text every 20 epochs to monitor quality
            if epoch % 20 == 0:
                try:
                    print(f"   📝 Generating sample at epoch {epoch}...")
                    if model is not None and preprocessed_data is not None:
                        model.eval()  # Set to evaluation mode
                        
                        # Create a simple generator for sampling
                        temp_generator = ModernGenerator(
                            model=model,
                            vocab_to_int=preprocessed_data["vocab_to_int"],
                            int_to_vocab=preprocessed_data["int_to_vocab"],
                            character_vocab=preprocessed_data["character_vocab"],
                            tokenizer=ModernTextProcessor(TOKENIZER_TYPE, MODEL_NAME).tokenizer,
                        )
                        
                        sample_text = temp_generator.generate_nucleus_sampling(
                            seed_text="<TYRION LANNISTER>",
                            max_length=100,
                            top_p=TOP_P,
                            temperature=TEMPERATURE,
                            character="tyrion lannister",
                            repetition_penalty=REPETITION_PENALTY
                        )
                        
                        print(f"   🎭 Sample: {sample_text[:150]}...")
                        logger.info(f"📝 SAMPLE AT EPOCH {epoch}: {sample_text}")
                        
                        model.train()  # Set back to training mode
                    else:
                        print(f"   ⚠️ Model or data not available for sampling")
                        logger.warning(f"⚠️ Cannot generate sample at epoch {epoch}: Model or data is None")
                except Exception as e:
                    print(f"   ⚠️ Could not generate sample: {e}")
                    logger.warning(f"⚠️ Sample generation failed at epoch {epoch}: {e}")
            
            print()
        
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        print("🎉 TRAINING COMPLETED!")
        print(f"   🏁 Final loss: {training_losses[-1]:.6f}")
        print(f"   🏆 Best loss: {best_loss:.6f}")
        print(f"   📈 Total improvement: {training_losses[0] - best_loss:.6f}")
        print(f"   ⏰ Total training time: {total_time_str}")
        
        # Final logging
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"🏁 Final loss: {training_losses[-1]:.6f}")
        logger.info(f"🏆 Best loss achieved: {best_loss:.6f}")
        logger.info(f"📈 Total improvement: {training_losses[0] - best_loss:.6f}")
        logger.info(f"⏰ Total training time: {total_time_str}")
        logger.info(f"📊 Average time per epoch: {total_time/num_epochs:.1f}s")
        
        return training_losses, best_loss
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("💾 Saving current progress...")
        logger.warning("⚠️ Training interrupted by user")
        save_modern_model(f"{MODEL_SAVE_PATH}_interrupted", model)
        logger.info(f"💾 Interrupted model saved to {MODEL_SAVE_PATH}_interrupted")
        return training_losses, best_loss
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        logger.error(f"❌ Training error: {e}")
        return training_losses, best_loss

# ================================================================
# 🚀 MAIN EXECUTION (Protected with multiprocessing guard)
# FOR NON-PROGRAMMERS: This prevents Windows multiprocessing issues
# ================================================================

if __name__ == '__main__':
    # Add multiprocessing freeze_support for Windows compatibility
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Run the analysis
    analysis_results = analyze_got_data()

    # Run preprocessing
    preprocessed_data = preprocess_got_data()

    if preprocessed_data is None:
        print("❌ Cannot continue without preprocessed data. Please fix the issues above.")
        exit(1)

    train_loader = prepare_training_data(preprocessed_data)

    if train_loader is None:
        print("❌ Cannot continue without training data. Please fix the issues above.")
        exit(1)

    model, vocab_size, characters = create_ai_model(preprocessed_data)

    if model is None:
        print("❌ Cannot continue without a valid model. Please fix the issues above.")
        exit(1)

    trainer = setup_trainer(model, train_loader, preprocessed_data)

    if trainer is None:
        print("❌ Cannot continue without a trainer. Please fix the issues above.")
        exit(1)

    # Start training
    training_losses, final_best_loss = train_ai_model(trainer, NUM_EPOCHS, model, preprocessed_data, vocab_size, characters)

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
        vocab_to_int=preprocessed_data["vocab_to_int"],
        int_to_vocab=preprocessed_data["int_to_vocab"],
        character_vocab=preprocessed_data["character_vocab"],
        tokenizer=ModernTextProcessor(TOKENIZER_TYPE, MODEL_NAME).tokenizer,
    )

    script = generator.generate_nucleus_sampling(
        seed_text="<TYRION LANNISTER>",
        max_length=200,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        character="tyrion lannister",
        repetition_penalty=REPETITION_PENALTY
    )
    print(script)

    print("\n✅ Script generation complete.")

    # ================================================================
    # 📊 CREATE TRAINING VISUALIZATIONS (Show progress graphs)
    # FOR NON-PROGRAMMERS: Make pretty charts showing how well our AI learned
    # ================================================================

    print("\n📊 Creating training progress visualizations...")
    try:
        from .modern_plot import quick_dashboard
        quick_dashboard()
        print("✅ Training visualizations created! Check the generated .html files")
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")

    print("\n" + "🏁 MAIN SCRIPT COMPLETE!" + "\n")
    print("=" * 60)
    print("🎉 Your Game of Thrones AI script generator is ready!")
    print()
    print("📁 Generated Files:")
    print("   🤖 Trained AI model")  
    print("   📊 Training visualizations")
    print("   📈 Performance graphs")
    print("   📝 Sample generated dialogue")
    print()
    print("🚀 Next Steps:")
    print("   1. Check the sample dialogue generated above")
    print("   2. Open the .html visualization files in your browser")  
    print("   3. Experiment with different character prompts")
    print("   4. Adjust creativity settings (temperature, top_p)")
    print("   5. Generate longer or shorter dialogue snippets")
    print()
    print("🎭 Have fun with your AI Game of Thrones writer! 🐉⚔️👑")
