# validate_pipe_fix.py
# Simple validation of the pipe token fix

def test_decode_function():
    """Simple test of the decode function"""
    from modern_example_usage import ModernGenerator
    
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
