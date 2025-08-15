# test_fixes.py
# Quick test to verify our multiprocessing fixes

import sys
import multiprocessing

def test_multiprocessing_fix():
    """Test that multiprocessing works correctly"""
    print("🧪 Testing multiprocessing fix...")
    
    try:
        # Test freeze_support
        multiprocessing.freeze_support()
        print("✅ multiprocessing.freeze_support() works")
        
        # Test DataLoader with num_workers=0 on Windows
        import platform
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        x = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))
        dataset = TensorDataset(x, y)
        
        # Set num_workers based on platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=num_workers
        )
        
        # Test iteration
        for batch_x, batch_y in dataloader:
            break  # Just test first batch
            
        print(f"✅ DataLoader with num_workers={num_workers} works on {platform.system()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    print("🔍 TESTING MULTIPROCESSING FIXES")
    print("=" * 40)
    
    success = test_multiprocessing_fix()
    
    if success:
        print("\n🎉 All multiprocessing fixes work correctly!")
        print("✅ main_modern.py should now run without Windows multiprocessing errors")
    else:
        print("\n❌ Some fixes may still need attention")
