# quick_generation_test.py
# Test generation with the trained model and pipe token fix

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch
from torch import nn
from got_script_generator.modern_example_usage import (
    ModernScriptRNN,
    ModernGenerator,
)
from got_script_generator.improved_helperAI import load_modern_preprocess, SPECIAL_WORDS


class _DummyModel(nn.Module):
    """Minimal model to deterministically output a punctuation placeholder."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # Register a parameter so `parameters()` yields a tensor with device info.
        self.dummy_weight = nn.Parameter(torch.zeros(1))

    def forward(self, input_seq, hidden, character_ids=None):
        batch_size, seq_len = input_seq.shape
        device = input_seq.device
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -float("inf"),
            device=device,
        )
        # Always prefer token index 1 so we can test custom token decoding.
        logits[:, -1, 1] = 10.0
        return logits, hidden

    def init_hidden(self, batch_size, device="cpu"):
        return (
            torch.zeros(1, batch_size, 1, device=device),
            torch.zeros(1, batch_size, 1, device=device),
        )


def test_generate_decodes_custom_tokens():
    """Ensure nucleus sampling converts custom punctuation tokens back to symbols."""

    vocab_to_int = {
        "hello": 0,
        SPECIAL_WORDS["UNKNOWN"]: 2,
    }
    int_to_vocab = {
        0: "hello",
        1: "||period||",
        2: SPECIAL_WORDS["UNKNOWN"],
    }

    generator = ModernGenerator(
        model=_DummyModel(vocab_size=3),
        vocab_to_int=vocab_to_int,
        int_to_vocab=int_to_vocab,
        character_vocab={},
        tokenizer=None,
    )

    torch.manual_seed(0)
    output = generator.generate_nucleus_sampling(
        seed_text="hello",
        max_length=1,
        top_p=0.9,
        temperature=1.0,
    )

    assert "||" not in output, "Custom token placeholders should be decoded"
    assert output.strip().endswith("."), "Decoded text should include punctuation symbol"

    print("✅ Custom token decoding test passed!")
    return True


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
    test_generate_decodes_custom_tokens()
    test_generation_with_fix()
