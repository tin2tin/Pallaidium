"""
MPS-compatible diffusers utilities for Apple Silicon Macs
Provides optimizations and workarounds for running diffusers models on Metal
"""

import torch
from .logger import logger, log_func_call

@log_func_call
def optimize_model_for_mps(pipe):
    """
    Apply optimizations to make diffusers models run better on MPS
    
    Args:
        pipe: The diffusers pipeline to optimize
        
    Returns:
        The optimized pipeline
    """
    # Skip if MPS is not available
    if not (hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return pipe
    
    try:
        # Move the pipeline to MPS device
        pipe = pipe.to("mps")
        
        # Enable memory-efficient attention if available
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing(1)
            logger.info("Enabled attention slicing for MPS")
        
        # Use float16 precision for performance on models that support it
        if hasattr(pipe, 'unet') and not str(pipe.unet.dtype).startswith('torch.float16'):
            try:
                # Some models may not support half precision, so we handle exceptions
                logger.info("Attempting to use float16 for better MPS performance")
                pipe.to(torch.float16)
            except Exception as e:
                logger.warning(f"Could not convert to float16: {e}")
        
        # Enable sequential CPU offload for large models to prevent OOM errors
        try:
            # Only do this for large models like SDXL
            if hasattr(pipe, 'unet') and hasattr(pipe.unet, 'config'):
                if getattr(pipe.unet.config, 'in_channels', 0) >= 16:  # Likely a large model
                    if hasattr(pipe, 'enable_model_cpu_offload'):
                        pipe.enable_model_cpu_offload()
                        logger.info("Enabled CPU offloading for large model")
        except Exception as e:
            logger.warning(f"Could not enable CPU offloading: {e}")
        
        return pipe
    
    except Exception as e:
        logger.error(f"Error optimizing model for MPS: {e}")
        # Return original pipeline if optimization fails
        return pipe

@log_func_call
def create_mps_generator(seed=None):
    """
    Create a generator compatible with MPS
    
    Args:
        seed (int, optional): Random seed
        
    Returns:
        torch.Generator: A generator compatible with the current device
    """
    if seed is None or seed == 0:
        return None
        
    try:
        if torch.backends.mps.is_available():
            # MPS generator seems to have issues with some models
            # For safety, we'll use CPU generator instead
            generator = torch.Generator().manual_seed(seed)
            logger.debug(f"Created CPU generator with seed {seed} for MPS compatibility")
            return generator
        elif torch.cuda.is_available():
            generator = torch.Generator("cuda").manual_seed(seed)
            return generator
        else:
            generator = torch.Generator().manual_seed(seed)
            return generator
    except Exception as e:
        logger.error(f"Error creating generator: {e}")
        # Return None if generator creation fails
        return None

@log_func_call
def check_mps_compatibility(model_name):
    """
    Check if a model is compatible with MPS
    
    Args:
        model_name (str): The model name to check
        
    Returns:
        tuple: (is_compatible, compatibility_notes)
    """
    # List of models known to have issues with MPS
    problematic_models = [
        "stabilityai/stable-diffusion-xl-base-1.0",  # Can cause memory issues
        "stabilityai/stable-video-diffusion-img2vid",  # Known to have MPS issues
        "runwayml/stable-diffusion-v1-5",  # May have issues with some operations
    ]
    
    # Models that are confirmed to work well with MPS
    compatible_models = [
        "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2-1",
    ]
    
    # Check if model is in problematic list
    if any(prob_model in model_name for prob_model in problematic_models):
        return False, "This model may have compatibility issues with MPS."
    
    # Check if model is in compatible list
    if any(compat_model in model_name for compat_model in compatible_models):
        return True, "This model is known to work with MPS."
    
    # For unknown models, return cautious compatibility
    return True, "Compatibility with MPS is unknown. Please report any issues."

@log_func_call
def mps_memory_cleanup():
    """
    Clean up MPS memory
    """
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Force garbage collection
        import gc
        gc.collect()
        
        # Empty cache if possible
        try:
            torch.mps.empty_cache()
            logger.info("MPS cache cleared")
        except:
            pass
