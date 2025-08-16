# test_pipe_fix.py
# Quick test to verify the pipe token fix is working

from modern_example_usage import ModernGenerator
from improved_helperAI import load_modern_preprocess, ModernTextProcessor
import torch

def test_pipe_token_fix():
    """Test if the decode_custom_tokens fix resolves the pipe issue"""
    print("🧪 TESTING PIPE TOKEN FIX")
    print("=" * 50)
    
    try:
        # Load the existing trained model and preprocessed data
        print("📁 Loading preprocessed data...")
        preprocessed_data = load_modern_preprocess("preprocess_modern.pkl")
        print(f"   ✅ Vocabulary size: {len(preprocessed_data['vocab_to_int'])}")
        
        # Check if pipe tokens exist in vocabulary
        vocab = preprocessed_data['vocab_to_int']
        pipe_tokens = [token for token in vocab.keys() if 'pipe' in token.lower()]
        print(f"   🔍 Pipe tokens found in vocab: {pipe_tokens}")
        
        # Test the decode_custom_tokens method directly
        print("\n🔧 Testing decode_custom_tokens method...")
        
        # Create a generator instance
        dummy_model = None  # We don't need the model for this test
        generator = ModernGenerator(
            model=dummy_model,
            vocab_to_int=preprocessed_data["vocab_to_int"],
            int_to_vocab=preprocessed_data["int_to_vocab"],
            character_vocab=preprocessed_data.get("character_vocab", {}),
            tokenizer=None  # Use custom tokenizer
        )
        
        # Test decode function with sample pipe-heavy text
        test_text = "||pipe|| ||pipe|| catelyn stark ||pipe|| ||pipe|| greater ||pipe|| ||pipe|| my son ||pipe|| ||pipe|| period ||pipe|| ||pipe||"
        decoded_text = generator.decode_custom_tokens(test_text)
        
        print(f"   📝 Original: {test_text}")
        print(f"   📝 Decoded:  {decoded_text}")
        
        # Check if decoding worked
        if "||pipe||" not in decoded_text and "|" in decoded_text:
            print("   ✅ Decode function working correctly!")
        else:
            print("   ❌ Decode function not working properly")
            
        # Test multiple tokens
        print("\n🔧 Testing multiple token types...")
        multi_test = "||less|| tyrion lannister ||greater|| ||apostrophe|| ||apostrophe|| the wine helps me think ||comma|| ||comma|| he said ||period|| ||period|| ||question|| ||question||"
        multi_decoded = generator.decode_custom_tokens(multi_test)
        
        print(f"   📝 Multi-token original: {multi_test}")
        print(f"   📝 Multi-token decoded:  {multi_decoded}")
        
        if "||" not in multi_decoded:
            print("   ✅ Multi-token decode working correctly!")
        else:
            print("   ❌ Multi-token decode still has issues")
            
        print(f"\n✅ Pipe token fix test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_pipe_token_fix()
