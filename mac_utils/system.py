"""
System utilities for Mac
Provides replacements for Windows-specific system functions used in Pallaidium
"""

import os
import platform
import subprocess
from pathlib import Path
import torch
from .logger import logger, log_func_call

@log_func_call
def show_mac_console(show=True):
    """
    Mac equivalent of the Windows show_system_console function.
    Opens Terminal.app if needed.
    
    Args:
        show (bool): Whether to show the console or not
    """
    if show:
        # AppleScript to check if Terminal is running and open it if not
        applescript = '''
        tell application "System Events"
            set isRunning to (name of processes) contains "Terminal"
        end tell
        if isRunning is false then
            tell application "Terminal"
                activate
            end tell
        end if
        '''
        try:
            subprocess.run(["osascript", "-e", applescript], check=True)
            logger.info("Activated Terminal.app")
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to open Terminal: {e}")

@log_func_call
def set_mac_console_topmost(top=True):
    """
    Mac equivalent of the Windows set_system_console_topmost function.
    Brings Terminal.app to front if needed.
    
    Args:
        top (bool): Whether to bring the console to the front
    """
    if top:
        # AppleScript to bring Terminal to front
        applescript = '''
        tell application "Terminal"
            activate
        end tell
        '''
        try:
            subprocess.run(["osascript", "-e", applescript], check=True)
            logger.info("Brought Terminal.app to front")
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to bring Terminal to front: {e}")

@log_func_call
def get_mac_memory_info():
    """
    Get memory information on macOS
    
    Returns:
        dict: Memory information including total, used, and available memory
    """
    try:
        # Use vm_stat to get memory information
        vm_stat_output = subprocess.check_output(['vm_stat'], universal_newlines=True)
        
        # Parse the output
        lines = vm_stat_output.split('\n')
        page_size = 4096  # Default page size on macOS
        
        memory_info = {}
        for line in lines:
            if 'page size of' in line:
                page_size = int(line.split('page size of ')[1].split(' bytes')[0])
            
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip().replace('.', '')
                if value_str.isdigit():
                    memory_info[key] = int(value_str) * page_size
        
        # Calculate total physical memory
        total_memory = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], universal_newlines=True)
        memory_info['total'] = int(total_memory.strip())
        
        return memory_info
    except Exception as e:
        logger.error(f"Error getting Mac memory info: {e}")
        return {}

@log_func_call
def get_mps_device_info():
    """
    Get information about the MPS device on Apple Silicon Macs
    
    Returns:
        dict: MPS device information
    """
    device_info = {
        'available': False,
        'device_name': 'Unknown',
        'memory_allocated': 0,
    }
    
    try:
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info['available'] = True
            
            # Try to get allocated memory
            try:
                device_info['memory_allocated'] = torch.mps.current_allocated_memory()
            except:
                pass
                
            # Try to determine device name from system info
            try:
                # Use system_profiler to get GPU info
                gpu_info = subprocess.check_output(
                    ['system_profiler', 'SPDisplaysDataType'], 
                    universal_newlines=True
                )
                
                # Parse the output to find the GPU name
                for line in gpu_info.split('\n'):
                    if 'Chipset Model:' in line:
                        device_info['device_name'] = line.split('Chipset Model:')[1].strip()
                        break
            except:
                pass
    except Exception as e:
        logger.error(f"Error getting MPS device info: {e}")
    
    return device_info

@log_func_call
def optimize_for_mps():
    """
    Apply optimizations for MPS device on Apple Silicon
    """
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Set environment variables for MPS optimization
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        logger.info("Applied MPS optimizations")
        return True
    return False
