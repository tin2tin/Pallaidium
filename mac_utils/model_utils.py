"""
Model utilities for Mac compatibility
Provides functions to help with loading and optimizing PyTorch models on MPS
"""

import os
import torch
import warnings
from .logger import logger, log_func_call

@log_func_call
def patch_model_for_mps(model):
    """
    Patch a PyTorch model for MPS compatibility
    
    Args:
        model: The PyTorch model to patch
        
    Returns:
        The patched model
    """
    if not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return model
    
    try:
        # Check if model is already on MPS
        if next(model.parameters()).device.type == 'mps':
            logger.debug("Model already on MPS device")
            return model
        
        # Move model to MPS
        logger.info("Moving model to MPS device")
        model = model.to('mps')
        
        return model
    except Exception as e:
        logger.error(f"Error patching model for MPS: {e}")
        return model

@log_func_call
def convert_tensor_device(tensor, target_device=None):
    """
    Safely convert a tensor to the target device, with fallback for MPS issues
    
    Args:
        tensor: The tensor to convert
        target_device: The target device (if None, will use MPS if available)
        
    Returns:
        The converted tensor
    """
    if tensor is None:
        return None
    
    # Determine target device
    if target_device is None:
        if torch.backends.mps.is_available():
            target_device = 'mps'
        elif torch.cuda.is_available():
            target_device = 'cuda'
        else:
            target_device = 'cpu'
    
    try:
        # If tensor is not a torch tensor, convert it
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        
        # Convert to target device
        if tensor.device.type != target_device:
            tensor = tensor.to(target_device)
        
        return tensor
    except Exception as e:
        logger.warning(f"Error converting tensor to {target_device}, falling back to CPU: {e}")
        # Fallback to CPU
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        return tensor.to('cpu')

@log_func_call
def optimize_inference_for_mps():
    """
    Apply global optimizations for MPS inference
    """
    if not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return
    
    # Set environment variables for better MPS performance
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Suppress common MPS warnings
    warnings.filterwarnings("ignore", message=".*MPS.*")
    warnings.filterwarnings("ignore", message=".*is a non-CUDA.*")
    
    # Log MPS optimization
    logger.info("Applied global optimizations for MPS inference")

@log_func_call
def get_optimal_batch_size(device_type, model_type="diffusion"):
    """
    Get the optimal batch size for the current device
    
    Args:
        device_type (str): The device type ('cuda', 'mps', or 'cpu')
        model_type (str): The type of model ('diffusion', 'clip', etc.)
        
    Returns:
        int: The optimal batch size
    """
    if device_type == 'mps':
        # Apple Silicon has unified memory, so we need to be conservative
        if model_type == "diffusion":
            return 1  # Most conservative for diffusion models
        elif model_type == "clip":
            return 4  # CLIP models are smaller
        else:
            return 2  # Default conservative batch size
    elif device_type == 'cuda':
        # CUDA devices usually have more VRAM
        if model_type == "diffusion":
            return 4
        elif model_type == "clip":
            return 16
        else:
            return 8
    else:  # CPU
        # CPU batch sizes should be small to avoid memory issues
        return 1

@log_func_call
def auto_free_memory(device_type='mps'):
    """
    Automatically free memory for the specified device
    
    Args:
        device_type (str): The device type to free memory for
    """
    import gc
    gc.collect()
    
    if device_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA memory cache cleared")
    elif device_type == 'mps' and torch.backends.mps.is_available():
        try:
            # Try to clear MPS cache (may not be available in all PyTorch versions)
            torch.mps.empty_cache()
            logger.debug("MPS memory cache cleared")
        except:
            logger.debug("MPS memory cache clearing not available")
    
    # Force another garbage collection
    gc.collect()
