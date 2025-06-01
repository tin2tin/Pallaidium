"""
Mac Utils Package for Pallaidium
Provides utilities for Apple Silicon and macOS compatibility
"""

from .logger import logger, setup_logger, log_func_call, log_mem_usage
from .system import (
    show_mac_console, 
    set_mac_console_topmost, 
    get_mac_memory_info, 
    get_mps_device_info,
    optimize_for_mps
)
from .diffusers_mps import (
    optimize_model_for_mps,
    create_mps_generator,
    check_mps_compatibility,
    mps_memory_cleanup
)
from .model_utils import (
    patch_model_for_mps,
    convert_tensor_device,
    optimize_inference_for_mps,
    get_optimal_batch_size,
    auto_free_memory
)
from .platform_compat import (
    OS_PLATFORM,
    IS_APPLE_SILICON,
    show_system_console,
    set_system_console_topmost,
    setup_platform_environment,
    get_platform_info
)

__all__ = [
    # Logger
    'logger',
    'setup_logger',
    'log_func_call',
    'log_mem_usage',
    
    # System
    'show_mac_console',
    'set_mac_console_topmost',
    'get_mac_memory_info',
    'get_mps_device_info',
    'optimize_for_mps',
    
    # Diffusers MPS
    'optimize_model_for_mps',
    'create_mps_generator',
    'check_mps_compatibility',
    'mps_memory_cleanup',
    
    # Model Utilities
    'patch_model_for_mps',
    'convert_tensor_device',
    'optimize_inference_for_mps',
    'get_optimal_batch_size',
    'auto_free_memory',
    
    # Platform Compatibility
    'OS_PLATFORM',
    'IS_APPLE_SILICON',
    'show_system_console',
    'set_system_console_topmost',
    'setup_platform_environment',
    'get_platform_info',
]
