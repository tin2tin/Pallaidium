"""
Platform compatibility utilities for Pallaidium
Provides unified interfaces for platform-specific operations
"""

import os
import platform
import ctypes
from .logger import logger, log_func_call
from .system import show_mac_console, set_mac_console_topmost

# Determine platform
OS_PLATFORM = platform.system()  # 'Linux', 'Darwin', 'Java', 'Windows'
IS_APPLE_SILICON = OS_PLATFORM == "Darwin" and platform.machine() == "arm64"

@log_func_call
def show_system_console(show):
    """
    Cross-platform function to show/hide system console
    
    Args:
        show (bool): Whether to show the console
    """
    if OS_PLATFORM == "Windows":
        # Windows implementation
        SW_HIDE = 0
        SW_SHOW = 5
        try:
            ctypes.windll.user32.ShowWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), SW_SHOW if show else SW_HIDE
            )
            logger.debug(f"Windows console {'shown' if show else 'hidden'}")
        except Exception as e:
            logger.error(f"Error showing/hiding Windows console: {e}")
    
    elif OS_PLATFORM == "Darwin":
        # macOS implementation
        show_mac_console(show)
    
    else:
        # Linux implementation (no-op for now)
        logger.debug(f"Show console not implemented for {OS_PLATFORM}")

@log_func_call
def set_system_console_topmost(top):
    """
    Cross-platform function to set console window as topmost
    
    Args:
        top (bool): Whether to set the console as topmost
    """
    if OS_PLATFORM == "Windows":
        # Windows implementation
        HWND_NOTOPMOST = -2
        HWND_TOPMOST = -1
        HWND_TOP = 0
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOZORDER = 0x0004
        try:
            ctypes.windll.user32.SetWindowPos(
                ctypes.windll.kernel32.GetConsoleWindow(),
                HWND_TOP if top else HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER,
            )
            logger.debug(f"Windows console set {'topmost' if top else 'not topmost'}")
        except Exception as e:
            logger.error(f"Error setting Windows console topmost: {e}")
    
    elif OS_PLATFORM == "Darwin":
        # macOS implementation
        set_mac_console_topmost(top)
    
    else:
        # Linux implementation (no-op for now)
        logger.debug(f"Set console topmost not implemented for {OS_PLATFORM}")

@log_func_call
def setup_platform_environment():
    """
    Set up platform-specific environment variables and configurations
    """
    if OS_PLATFORM == "Darwin":
        # macOS specific environment setup
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        if IS_APPLE_SILICON:
            # Apple Silicon specific optimizations
            logger.info("Setting up environment for Apple Silicon")
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
    
    elif OS_PLATFORM == "Windows":
        # Windows specific environment setup
        pass
    
    else:
        # Linux specific environment setup
        pass
    
    # Common environment setup
    os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is unbuffered

@log_func_call
def get_platform_info():
    """
    Get platform-specific information
    
    Returns:
        dict: Platform information
    """
    info = {
        "platform": OS_PLATFORM,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "is_apple_silicon": IS_APPLE_SILICON,
    }
    
    # Add more detailed platform information
    if OS_PLATFORM == "Darwin":
        info["macos_version"] = platform.mac_ver()[0]
        
        # Check for MPS support
        try:
            import torch
            info["mps_available"] = (
                hasattr(torch, 'backends') and 
                hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available()
            )
        except ImportError:
            info["mps_available"] = False
    
    elif OS_PLATFORM == "Windows":
        info["windows_version"] = platform.win32_ver()[0]
        
        # Check for CUDA support
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["cuda_devices"] = []
                for i in range(torch.cuda.device_count()):
                    info["cuda_devices"].append({
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i)
                    })
        except ImportError:
            info["cuda_available"] = False
    
    return info
