# debug.py
# 🐛 DEBUG AND DIAGNOSTICS TOOL
# 
# FOR NON-PROGRAMMERS:
# This file helps you troubleshoot problems with your Game of Thrones AI.
# Think of it as a "doctor" that examines your system and tells you what's wrong.
# Run this whenever you encounter issues or want to check if everything is working.
#
# Comprehensive debugging and system diagnostics for Game of Thrones AI Script Generator

import sys
import os
import json
import pickle
import traceback
from datetime import datetime
import platform
import subprocess
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ================================================================
# 🎨 Colors for Terminal Output
# ================================================================
class Colors:
    """Terminal color codes for better readability"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors for compatibility"""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ''
        cls.PURPLE = cls.CYAN = cls.WHITE = cls.BOLD = ''
        cls.UNDERLINE = cls.END = ''

# Disable colors on Windows if needed
if platform.system() == 'Windows' and not os.environ.get('ANSICON'):
    Colors.disable()

# ================================================================
# 🛠️ Utility Functions
# ================================================================

def print_header(title):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}{Colors.END}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def safe_import(module_name, package_name=None):
    """Safely import a module and return status"""
    try:
        if package_name:
            __import__(package_name)
            module = sys.modules[package_name]
            version = getattr(module, '__version__', 'Unknown')
            return True, version
        else:
            __import__(module_name)
            module = sys.modules[module_name]
            version = getattr(module, '__version__', 'Unknown')
            return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_command(command):
    """Safely run a command and return output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_file_exists(filepath, description="File"):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        if size > 0:
            print_success(f"{description}: {filepath} ({size:,} bytes)")
            return True
        else:
            print_warning(f"{description}: {filepath} (empty file)")
            return False
    else:
        print_error(f"{description}: {filepath} (not found)")
        return False

# ================================================================
# 🖥️ SYSTEM DIAGNOSTICS
# ================================================================

def check_system_info():
    """Check basic system information"""
    print_header("SYSTEM INFORMATION")
    
    print_info(f"Operating System: {platform.system()} {platform.release()}")
    print_info(f"Architecture: {platform.machine()}")
    print_info(f"Processor: {platform.processor()}")
    print_info(f"Python Version: {sys.version}")
    print_info(f"Python Executable: {sys.executable}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print_info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
        print_info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        print_info(f"RAM Usage: {memory.percent}%")
    except ImportError:
        print_warning("psutil not installed - cannot check memory info")
    except Exception as e:
        print_warning(f"Could not retrieve memory info: {e}")

def check_python_environment():
    """Check Python environment and virtual environment"""
    print_header("PYTHON ENVIRONMENT")
    
    # Check if in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Running in virtual environment")
        print_info(f"Virtual environment: {sys.prefix}")
    else:
        print_warning("Not running in virtual environment (not recommended)")
    
    # Check pip
    success, stdout, stderr = run_command("pip --version")
    if success:
        print_success(f"pip: {stdout}")
    else:
        print_error(f"pip not available: {stderr}")
    
    # Check Python version compatibility
    version_info = sys.version_info
    if version_info >= (3, 10) and version_info < (3, 12):
        print_success(f"Python version {version_info.major}.{version_info.minor} is compatible")
    else:
        print_warning(f"Python version {version_info.major}.{version_info.minor} may have compatibility issues")

# ================================================================
# 📦 PACKAGE DIAGNOSTICS
# ================================================================

def check_core_packages():
    """Check installation of core packages"""
    print_header("CORE PACKAGE INSTALLATION")
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'plotly': 'Plotly',
        'datasets': 'Hugging Face Datasets',
        'sentencepiece': 'SentencePiece'
    }
    
    all_good = True
    for package, name in packages.items():
        success, version = safe_import(package)
        if success:
            print_success(f"{name}: {version}")
        else:
            print_error(f"{name}: {version}")
            all_good = False
    
    return all_good

def check_pytorch_cuda():
    """Check PyTorch CUDA installation and GPU availability"""
    print_header("PYTORCH & CUDA STATUS")
    
    try:
        import torch
        print_success(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print_success("CUDA is available")
            
            # Get basic CUDA info safely
            try:
                # Show compute capability which is more reliable
                if torch.cuda.device_count() > 0:
                    capability = torch.cuda.get_device_capability(0)
                    print_info(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
                
                # Check if we can get CUDA version from environment
                import subprocess
                try:
                    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and 'release' in result.stdout:
                        version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
                        if version_line:
                            print_info(f"NVIDIA CUDA: {version_line[0].strip()}")
                    else:
                        print_info("CUDA: Available (nvcc not in PATH)")
                except:
                    print_info("CUDA: Available (version detection unavailable)")
                    
            except Exception as e:
                print_info(f"CUDA: Available (info gathering failed)")
            
            # Get cuDNN version safely
            try:
                if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                    if hasattr(torch.backends.cudnn, 'version') and callable(torch.backends.cudnn.version):
                        cudnn_version = torch.backends.cudnn.version()
                        print_info(f"cuDNN version: {cudnn_version}")
                    else:
                        print_info("cuDNN: Available")
                else:
                    print_info("cuDNN: Not available")
            except Exception as e:
                print_info("cuDNN: Status unknown")
            
            # GPU information
            gpu_count = torch.cuda.device_count()
            print_info(f"GPU count: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print_info(f"GPU {i}: {props.name}")
                print_info(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
                print_info(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Test CUDA functionality
            try:
                x = torch.tensor([1.0]).cuda()
                y = torch.tensor([2.0]).cuda()
                z = x + y
                print_success("CUDA functionality test passed")
            except Exception as e:
                print_error(f"CUDA functionality test failed: {e}")
        else:
            print_warning("CUDA not available - will use CPU")
            print_info("For GPU acceleration, ensure NVIDIA drivers and CUDA are installed")
        
        return True
    except ImportError:
        print_error("PyTorch not installed")
        return False

# ================================================================
# 📁 PROJECT FILES CHECK
# ================================================================

def check_project_files():
    """Check if all required project files exist"""
    print_header("PROJECT FILES")
    
    files_status = {}
    
    # Core Python files
    core_files = [
        ("main_modern.py", "Main training script"),
        ("modern_example_usage.py", "Enhanced training script"),
        ("improved_helperAI.py", "Core helper functions"),
        ("modern_plot.py", "Visualization tools"),
        ("requirements.txt", "Dependencies list")
    ]
    
    for filepath, description in core_files:
        files_status[filepath] = check_file_exists(filepath, description)
    
    # Data directory
    if os.path.exists("data"):
        print_success("Data directory exists")
        data_files = os.listdir("data")
        if data_files:
            print_info(f"Data files found: {', '.join(data_files)}")
        else:
            print_warning("Data directory is empty")
        files_status["data"] = len(data_files) > 0
    else:
        print_error("Data directory not found")
        files_status["data"] = False
    
    # Check for specific data file
    got_data_exists = check_file_exists("data/Game_of_Thrones_Script.csv", "Game of Thrones dataset")
    files_status["got_data"] = got_data_exists
    
    return files_status

def check_generated_files():
    """Check for generated files from previous runs"""
    print_header("GENERATED FILES")
    
    patterns = [
        ("*.pt", "PyTorch model files"),
        ("*.pkl", "Preprocessed data files"),
        ("*.html", "Visualization files"),
        ("training_output.txt", "Training logs")
    ]
    
    for pattern, description in patterns:
        if pattern == "training_output.txt":
            if os.path.exists(pattern):
                size = os.path.getsize(pattern)
                print_success(f"{description}: {pattern} ({size:,} bytes)")
            else:
                print_info(f"{description}: Not found (normal for first run)")
        else:
            import glob
            files = glob.glob(pattern)
            if files:
                total_size = sum(os.path.getsize(f) for f in files)
                print_success(f"{description}: {len(files)} files ({total_size:,} bytes total)")
                for f in files[:3]:  # Show first 3 files
                    print_info(f"  - {f}")
                if len(files) > 3:
                    print_info(f"  ... and {len(files) - 3} more")
            else:
                print_info(f"{description}: None found (normal for first run)")

def check_preprocessed_data():
    """Check preprocessed data file in detail"""
    print_header("PREPROCESSED DATA ANALYSIS")
    
    pkl_file = "preprocess_modern.pkl"
    if os.path.exists(pkl_file):
        try:
            data = pickle.load(open(pkl_file, "rb"))
            print_success(f"Successfully loaded {pkl_file}")
            
            print_info(f"Vocab size: {len(data.get('vocab_to_int', {}))}")
            print_info(f"Number of sequences: {len(data.get('sequences', []))}")
            
            if 'metadata' in data:
                metadata = data['metadata']
                print_info(f"Characters: {len(metadata.get('characters', []))}")
                print_info(f"Context window: {metadata.get('window', 'Unknown')}")
                
            # Sample data
            if 'int_to_vocab' in data:
                sample_vocab = list(data['int_to_vocab'].items())[:10]
                print_info(f"Sample vocabulary: {sample_vocab}")
                
            if 'sequences' in data and len(data['sequences']) > 0:
                print_info(f"First sequence shape: {len(data['sequences'][0])}")
                print_info(f"First few tokens: {data['sequences'][0][:10]}")
                
        except Exception as e:
            print_error(f"Error loading preprocessed data: {e}")
    else:
        print_warning("No preprocessed data found - run preprocessing first")

# ================================================================
# 🧪 FUNCTIONALITY TESTS
# ================================================================

def test_imports():
    """Test importing our custom modules"""
    print_header("MODULE IMPORT TESTS")
    
    modules = [
        ("improved_helperAI", "Core helper functions"),
        ("modern_example_usage", "Training and generation classes"),
        ("modern_plot", "Plotting utilities")
    ]
    
    all_imported = True
    for module_name, description in modules:
        try:
            __import__(module_name)
            print_success(f"{description}: {module_name}")
        except ImportError as e:
            print_error(f"{description}: {module_name} - {e}")
            all_imported = False
        except Exception as e:
            print_error(f"{description}: {module_name} - Unexpected error: {e}")
            all_imported = False
    
    return all_imported

def test_data_loading():
    """Test data loading functionality"""
    print_header("DATA LOADING TEST")
    
    try:
        from got_script_generator.improved_helperAI import analyze_dataset
        
        data_file = "data/Game_of_Thrones_Script.csv"
        if not os.path.exists(data_file):
            print_warning("Game of Thrones dataset not found - skipping data test")
            return False
        
        print_info("Analyzing dataset...")
        result = analyze_dataset(data_file)
        
        if result:
            print_success("Dataset analysis completed")
            print_info(f"Total dialogues: {result.get('total_dialogues', 'Unknown')}")
            print_info(f"Unique characters: {result.get('unique_characters', 'Unknown')}")
            print_info(f"Vocabulary size: {result.get('vocabulary_size', 'Unknown')}")
            return True
        else:
            print_error("Dataset analysis failed")
            return False
            
    except Exception as e:
        print_error(f"Data loading test failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation functionality"""
    print_header("MODEL CREATION TEST")
    
    try:
        from got_script_generator.modern_example_usage import ModernScriptRNN
        print_info("Testing model creation...")
        
        # Create a small test model
        model = ModernScriptRNN(
            vocab_size=1000,
            output_size=1000,
            embedding_dim=128,
            hidden_dim=256,
            n_layers=2,
            characters=["test_character"],
            dropout=0.3
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print_success(f"Model created successfully with {total_params:,} parameters")
        
        # Test model forward pass
        import torch
        test_input = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        
        with torch.no_grad():
            # Initialize hidden state for the model
            batch_size = test_input.size(0)
            hidden = model.init_hidden(batch_size)
            
            # Forward pass with both input and hidden state
            output, hidden = model(test_input, hidden)
            print_success(f"Forward pass successful - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print_error(f"Model creation test failed: {e}")
        traceback.print_exc()
        return False

# ================================================================
# 🏥 HEALTH CHECK SUMMARY
# ================================================================

def run_health_check():
    """Run complete health check and provide summary"""
    print_header("HEALTH CHECK SUMMARY")
    
    checks = {
        "System Info": True,  # Always passes
        "Python Environment": True,  # Always passes  
        "Core Packages": check_core_packages(),
        "PyTorch & CUDA": check_pytorch_cuda(),
        "Project Files": all(check_project_files().values()),
        "Module Imports": test_imports(),
        "Data Loading": test_data_loading(),
        "Model Creation": test_model_creation()
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    print_info(f"Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print_success("🎉 All systems operational! Your Game of Thrones AI is ready!")
        print_info("You can now run: python main_modern.py")
    else:
        print_warning("⚠️ Some issues detected. See details above.")
        
        # Provide specific recommendations
        if not checks["Core Packages"]:
            print_info("💡 Install missing packages: pip install -r requirements.txt")
        
        if not checks["PyTorch & CUDA"]:
            print_info("💡 For GPU support, install CUDA and compatible PyTorch version")
        
        if not checks["Project Files"]:
            print_info("💡 Ensure you have all project files and the Game of Thrones dataset")
        
        if not checks["Data Loading"]:
            print_info("💡 Check that data/Game_of_Thrones_Script.csv exists and is valid")

# ================================================================
# 🚨 TROUBLESHOOTING GUIDE
# ================================================================

def print_troubleshooting_guide():
    """Print common troubleshooting solutions"""
    print_header("TROUBLESHOOTING GUIDE")
    
    print(f"{Colors.BOLD}Common Issues and Solutions:{Colors.END}")
    print()
    
    issues = [
        ("CUDA out of memory", [
            "Reduce BATCH_SIZE in main_modern.py (try 8 or 4)",
            "Reduce CONTEXT_WINDOW (try 32 or 16)",
            "Close other GPU applications",
            "Use CPU-only training: run_project.bat --cpu"
        ]),
        ("Module not found errors", [
            "Activate virtual environment: venv\\Scripts\\activate",
            "Install requirements: pip install -r requirements.txt",
            "Check Python path and working directory"
        ]),
        ("Training is very slow", [
            "Install CUDA version of PyTorch for GPU acceleration",
            "Update NVIDIA drivers",
            "Increase BATCH_SIZE if you have enough GPU memory",
            "Use SSD storage for faster data loading"
        ]),
        ("Model generates nonsense", [
            "Train for more epochs (increase NUM_EPOCHS)",
            "Check that your dataset is properly formatted",
            "Verify preprocessing completed successfully",
            "Try different temperature settings (0.7-1.0)"
        ]),
        ("Data loading errors", [
            "Ensure data/Game_of_Thrones_Script.csv exists",
            "Check CSV format (should have 'Character' and 'Line' columns)",
            "Verify file encoding (should be UTF-8)",
            "Check file permissions"
        ])
    ]
    
    for i, (issue, solutions) in enumerate(issues, 1):
        print(f"{Colors.YELLOW}{i}. {issue}:{Colors.END}")
        for solution in solutions:
            print(f"   • {solution}")
        print()

# ================================================================
# 🔧 QUICK FIXES
# ================================================================

def offer_quick_fixes():
    """Offer to run quick fixes for common issues"""
    print_header("QUICK FIXES")
    
    fixes = [
        ("Install missing packages", "pip install -r requirements.txt"),
        ("Upgrade pip", "python -m pip install --upgrade pip"),
        ("Check GPU memory", "nvidia-smi"),
        ("Test PyTorch installation", "python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
    ]
    
    print("Available quick fixes:")
    for i, (description, command) in enumerate(fixes, 1):
        print(f"{i}. {description}")
        print(f"   Command: {Colors.CYAN}{command}{Colors.END}")
    
    print("\nTo run a fix, copy and paste the command into your terminal.")

# ================================================================
# 📊 PERFORMANCE BENCHMARK
# ================================================================

def run_performance_benchmark():
    """Run a quick performance benchmark"""
    print_header("PERFORMANCE BENCHMARK")
    
    try:
        import torch
        import time
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_info(f"Running benchmark on: {device}")
        
        # Matrix multiplication benchmark
        size = 1000
        print_info(f"Matrix multiplication benchmark ({size}x{size})...")
        
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warm up
        for _ in range(5):
            _ = torch.matmul(a, b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            _ = torch.matmul(a, b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print_success(f"Average time per operation: {avg_time*1000:.2f} ms")
        
        # Performance rating
        if device.type == 'cuda':
            if avg_time < 0.01:
                print_success("🚀 Excellent performance! Training should be fast.")
            elif avg_time < 0.05:
                print_success("✅ Good performance for training.")
            else:
                print_warning("⚠️ Slower performance - training may take longer.")
        else:
            print_info("💻 CPU performance - consider GPU for faster training.")
            
    except Exception as e:
        print_error(f"Benchmark failed: {e}")

# ================================================================
# 🎯 MAIN EXECUTION
# ================================================================

def main():
    """Main debug function"""
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("🐉⚔️👑 GAME OF THRONES AI - DEBUG & DIAGNOSTICS 🐺❄️")
    print("====================================================")
    print(f"Debug session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.END}")
    
    try:
        # Run all diagnostic checks
        check_system_info()
        check_python_environment()
        check_core_packages()
        check_pytorch_cuda()
        check_project_files()
        check_generated_files()
        check_preprocessed_data()
        test_imports()
        test_data_loading()
        test_model_creation()
        
        # Performance benchmark
        run_performance_benchmark()
        
        # Health check summary
        run_health_check()
        
        # Troubleshooting guide
        print_troubleshooting_guide()
        
        # Quick fixes
        offer_quick_fixes()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Debug session interrupted by user{Colors.END}")
    except Exception as e:
        print_error(f"Unexpected error during diagnostics: {e}")
        traceback.print_exc()
    
    print(f"\n{Colors.PURPLE}🏁 Debug session completed{Colors.END}")
    print("For more help, check the documentation or run specific tests.")

if __name__ == "__main__":
    main()
