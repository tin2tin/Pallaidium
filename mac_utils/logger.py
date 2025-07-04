"""
Pallaidium-4Mac Logger
Enhanced logging system for debugging Pallaidium on macOS/Apple Silicon
"""

import os
import logging
import inspect
import datetime
import platform
from pathlib import Path
import torch

# Configure logging levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Create logger
logger = logging.getLogger("Pallaidium-4Mac")

def setup_logger(log_level="INFO", log_to_file=True):
    """
    Set up the logger with appropriate handlers and formatting.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file (bool): Whether to save logs to a file
    """
    # Set the log level
    level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        logs_dir = Path(os.path.expanduser("~")) / ".pallaidium" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f"pallaidium_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    # Log initial system information
    log_system_info()
    
    return logger

def log_system_info():
    """Log system information for debugging purposes"""
    logger.info("=" * 50)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 50)
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Machine: {platform.machine()}")
    
    # Log torch information if available
    try:
        logger.info(f"PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info("CUDA: Available")
            logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA: Not available")
            
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS: Available")
            # Try to get more MPS info when possible
            try:
                logger.info(f"MPS Device: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB allocated")
            except:
                logger.info("MPS Device: Details unavailable")
        else:
            logger.info("MPS: Not available")
            
    except Exception as e:
        logger.warning(f"Could not log PyTorch information: {e}")
    
    logger.info("=" * 50)

def log_func_call(func):
    """Decorator to log function calls with parameters and execution time"""
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        module = inspect.getmodule(func).__name__
        
        # Format the arguments
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        
        # Log function entry
        logger.debug(f"CALL: {module}.{func_name}({signature})")
        
        # Measure execution time
        start_time = datetime.datetime.now()
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            logger.debug(f"RETURN: {module}.{func_name} completed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            logger.error(f"ERROR: {module}.{func_name} failed after {execution_time:.4f}s - {str(e)}")
            raise
    
    return wrapper

def log_mem_usage():
    """Log current memory usage"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"CUDA Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / (1024**3)
                logger.info(f"MPS Memory: {allocated:.2f} GB allocated")
            except:
                logger.info("MPS Memory: Unable to retrieve information")
    except Exception as e:
        logger.warning(f"Error logging memory usage: {e}")
