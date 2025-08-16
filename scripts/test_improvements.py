"""
Quick test script to validate the improvements made to the AI training system.
This will run a short training session to test the new configuration.
"""

import sys
sys.path.append('.')

from main_modern import *
from modern_example_usage import ModernGenerator
from improved_helperAI import ModernTextProcessor

def quick_test():
    print("🧪 TESTING IMPROVED CONFIGURATION")
    print("=" * 50)
    print(f"📊 New Settings:")
    print(f"   🎯 Tokenizer: {TOKENIZER_TYPE} (was: gpt2)")  
    print(f"   📚 Max Vocab: {MAX_VOCAB_SIZE} (was: 10000)")
    print(f"   📏 Context Window: {CONTEXT_WINDOW} (was: 64)")
    print(f"   🎓 Batch Size: {BATCH_SIZE} (was: 16)")
    print(f"   🧠 Model Dims: {EMBEDDING_DIM}/{HIDDEN_DIM} (was: 256/512)")
    print(f"   🎭 Temperature: {TEMPERATURE} (was: 0.8)")
    print(f"   🔄 Repetition Penalty: {REPETITION_PENALTY} (new)")
    print()

    # Test preprocessing
    print("1️⃣ Testing preprocessing...")
    preprocessed_data = preprocess_got_data()
    if preprocessed_data:
        print(f"   ✅ Vocabulary size: {len(preprocessed_data['vocab_to_int'])}")
        print(f"   ✅ Sequences: {len(preprocessed_data['sequences'])}")
    else:
        print("   ❌ Preprocessing failed!")
        return

    # Test data preparation
    print("\n2️⃣ Testing data preparation...")
    train_loader = prepare_training_data(preprocessed_data)
    if train_loader:
        print(f"   ✅ Training batches: {len(train_loader)}")
    else:
        print("   ❌ Data preparation failed!")
        return

    # Test model creation
    print("\n3️⃣ Testing model creation...")
    model, vocab_size, characters = create_ai_model(preprocessed_data)
    if model:
        print(f"   ✅ Model created with {vocab_size} vocab size")
    else:
        print("   ❌ Model creation failed!")
        return

    # Test generator
    print("\n4️⃣ Testing generator...")
    try:
        generator = ModernGenerator(
            model=model,
            vocab_to_int=preprocessed_data["vocab_to_int"],
            int_to_vocab=preprocessed_data["int_to_vocab"],
            character_vocab=preprocessed_data["character_vocab"],
            tokenizer=ModernTextProcessor(TOKENIZER_TYPE, MODEL_NAME).tokenizer,
        )
        print("   ✅ Generator created successfully")
    except Exception as e:
        print(f"   ❌ Generator creation failed: {e}")
        return
    
    # Test generation with new parameters
    print("\n5️⃣ Testing generation with new parameters...")
    try:
        # Test with proper character tag format
        sample = generator.generate_nucleus_sampling(
            seed_text="<TYRION LANNISTER>",
            max_length=80,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            character="tyrion lannister",
            repetition_penalty=REPETITION_PENALTY
        )
        print(f"   📝 Sample 1: {sample}")
        
        # Test with another character
        sample2 = generator.generate_nucleus_sampling(
            seed_text="<JON SNOW>",
            max_length=80,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            character="jon snow",
            repetition_penalty=REPETITION_PENALTY
        )
        print(f"   📝 Sample 2: {sample2}")
        
    except Exception as e:
        print(f"   ❌ Generation error: {e}")

    print(f"\n✅ Configuration test completed!")

if __name__ == "__main__":
    quick_test()
