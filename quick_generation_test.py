# quick_generation_test.py
# Test generation with the trained model and pipe token fix

import torch
from modern_example_usage import ModernScriptRNN, ModernGenerator
from improved_helperAI import load_modern_preprocess

def test_generation_with_fix():
    """Test if trained model + decode fix produces readable output"""
    print("🎬 TESTING GENERATION WITH PIPE TOKEN FIX")
    print("=" * 60)
    
    try:
        # Load preprocessed data
        print("📁 Loading preprocessed data...")
        preprocessed_data = load_modern_preprocess("preprocess_modern.pkl")
        vocab_size = len(preprocessed_data['vocab_to_int'])
        characters = list(preprocessed_data.get('character_vocab', {}).keys())
        
        print(f"   ✅ Vocabulary size: {vocab_size}")
        print(f"   ✅ Characters: {len(characters)} found")
        
        # Try to load the trained model
        print("\n🧠 Loading trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model with same architecture as training
        model = ModernScriptRNN(
            vocab_size=vocab_size,
            output_size=vocab_size,
            embedding_dim=384,  # From main_modern.py config
            hidden_dim=768,
            n_layers=3,
            characters=characters,
            dropout=0.2,
        ).to(device)
        
        # Try to load weights
        try:
            model.load_state_dict(torch.load('modern_script_model.pt', map_location=device))
            print("   ✅ Loaded trained model weights")
        except FileNotFoundError:
            print("   ⚠️  No trained model found, using random weights for testing")
        
        # Create generator with decode fix
        print("\n🎭 Creating generator with decode fix...")
        generator = ModernGenerator(
            model=model,
            vocab_to_int=preprocessed_data["vocab_to_int"],
            int_to_vocab=preprocessed_data["int_to_vocab"],
            character_vocab=preprocessed_data.get("character_vocab", {}),
            tokenizer=None  # Use custom tokenizer with decode fix
        )
        print("   ✅ Generator created with decode_custom_tokens method")
        
        # Test generation
        print("\n🗣️  Testing generation with different characters...")
        
        test_prompts = [
            "<TYRION LANNISTER>",
            "<JON SNOW>", 
            "<CATELYN STARK>",
            "<DAENERYS TARGARYEN>"
        ]
        
        for i, prompt in enumerate(test_prompts[:2]):  # Test first 2
            print(f"\n--- Test {i+1}: {prompt} ---")
            try:
                sample = generator.generate_nucleus_sampling(
                    seed_text=prompt,
                    max_length=100,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.2
                )
                
                print(f"📝 Generated: {sample[:200]}...")
                
                # Check if fix worked
                if "||pipe||" in sample:
                    print("   ❌ Still contains pipe tokens - fix not working")
                else:
                    print("   ✅ No pipe tokens found - fix working!")
                    
                if any(char.lower() in sample.lower() for char in ["tyrion", "jon", "catelyn", "daenerys"]):
                    print("   ✅ Character names detected")
                else:
                    print("   ⚠️  No clear character names in output")
                    
            except Exception as e:
                print(f"   ❌ Generation error: {e}")
        
        print(f"\n✅ Generation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_generation_with_fix()
