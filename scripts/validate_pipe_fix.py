# validate_pipe_fix.py
# Simple validation of the pipe token fix

def test_decode_function():
    """Simple test of the decode function"""
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC_PATH = ROOT / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

    from got_script_generator.modern_example_usage import ModernGenerator
    
    # Create a minimal generator just to test decode function
    generator = ModernGenerator(
        model=None,
        vocab_to_int={},
        int_to_vocab={},
        character_vocab={},
        tokenizer=None
    )
    
    # Test the decode function
    test_input = "||pipe|| ||pipe|| catelyn stark ||pipe|| ||greater|| my son ||pipe|| ||period||"
    decoded = generator.decode_custom_tokens(test_input)
    
    print("🧪 PIPE TOKEN DECODE TEST")
    print("=" * 40)
    print(f"Input:  {test_input}")
    print(f"Output: {decoded}")
    
    if "||pipe||" not in decoded and "|" in decoded:
        print("✅ SUCCESS: Pipe tokens decoded correctly!")
        return True
    else:
        print("❌ FAILED: Pipe tokens not decoded properly")
        return False

if __name__ == "__main__":
    test_decode_function()
