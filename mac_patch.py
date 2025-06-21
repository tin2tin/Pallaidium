"""
Pallaidium Mac Patch
Applies patches to make Pallaidium work on macOS, especially Apple Silicon

This script should be imported in the main __init__.py file before other imports
"""

import os
import sys
import platform
import pathlib

# Initialize basic platform detection
os_platform = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'
is_apple_silicon = os_platform == "Darwin" and platform.machine() == "arm64"

# Fix path handling for different platforms
if os_platform == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
elif os_platform == "Darwin":
    # Ensure we don't accidentally use Windows paths on Mac
    pass  # PosixPath is default on Mac

# Set environment variables for better MPS performance
if os_platform == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Disable oneDNN optimizations that can cause issues on Apple Silicon
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable certain warnings that are common with PyTorch on Apple Silicon
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="The operator.*is not current")
warnings.filterwarnings("ignore", category=UserWarning, message="Converting a tensor to a Python boolean")

# Make sure site-packages is in the path
python_exe_dir = os.path.dirname(sys.__file__)
site_packages_dir = os.path.join(python_exe_dir, "lib", "site-packages")
if site_packages_dir not in sys.path:
    sys.path.insert(0, site_packages_dir)

# Setup basic console logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Pallaidium-Mac")

logger.info(f"Pallaidium Mac Patch initialized")
logger.info(f"Platform: {os_platform}")
logger.info(f"Is Apple Silicon: {is_apple_silicon}")

# Print Python path for debugging
logger.debug(f"Python Path: {sys.path}")

# Try to import torch to check MPS availability
try:
    import torch
    if os_platform == "Darwin" and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            logger.info("MPS is available - will use for GPU acceleration")
        else:
            logger.warning("MPS is not available - will use CPU only")
except ImportError:
    logger.warning("PyTorch not installed yet")
except Exception as e:
    logger.error(f"Error checking MPS availability: {e}")

# Apply monkey patches for compatibility
def apply_patches():
    """Apply monkey patches to make things work on Mac"""
    if os_platform != "Darwin":
        return
    
    logger.info("Applying Mac compatibility patches")
    
    # Patch torch CUDA functions if needed
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        
        # Only patch if we're on Mac without CUDA
        if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
            # We need to ensure cuda functions don't crash when called on Mac
            # Many parts of the code assume cuda.is_available() or call cuda functions
            
            # Original implementation handles this gracefully, but let's add extra safety
            original_cuda_is_available = torch.cuda.is_available
            
            def patched_cuda_is_available():
                # On Mac, we need to ensure this doesn't error out
                try:
                    return original_cuda_is_available()
                except:
                    return False
            
            # Apply the patch
            torch.cuda.is_available = patched_cuda_is_available
            logger.info("Patched torch.cuda.is_available for Mac compatibility")

# Apply the patches
apply_patches()
