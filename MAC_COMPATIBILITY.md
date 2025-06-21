# Pallaidium for macOS

This document explains the modifications and additions made to port Pallaidium from Windows to macOS, with specific optimizations for Apple Silicon (M-series) Macs.

## Overview

Pallaidium has been enhanced with macOS compatibility, focusing on:

1. **Cross-platform compatibility** - Replacing Windows-specific system functions with macOS equivalents
2. **Apple Silicon support** - Enabling GPU acceleration via Metal Performance Shaders (MPS)
3. **Memory optimization** - Adjusting for the unified memory architecture of Apple Silicon
4. **Enhanced debugging** - Adding robust logging for easier troubleshooting
5. **Dependency management** - Ensuring proper installation of Apple Silicon native packages

## Mac-specific Modules

### 1. Core Mac Patch (`mac_patch.py`)

The early initialization module applies critical patches before other imports:

- Sets up environment variables for MPS optimization
- Fixes path handling for macOS
- Suppresses common PyTorch MPS warnings
- Provides early detection of Apple Silicon hardware

### 2. Platform Compatibility (`mac_utils/platform_compat.py`)

Provides cross-platform abstractions:

- Unified interfaces for system console management
- Platform detection utilities
- Environment setup for each platform (Windows, macOS, Linux)
- Special handling for Apple Silicon Macs

### 3. MPS-specific Diffusers Utilities (`mac_utils/diffusers_mps.py`)

Optimizes diffusion models for Metal:

- Model optimizations for MPS devices
- MPS-compatible random generators
- Model compatibility checking for known issues
- Memory management for Metal devices

### 4. PyTorch Model Utilities (`mac_utils/model_utils.py`)

General model utilities for macOS:

- PyTorch model patching for MPS compatibility
- Tensor device conversion with fallbacks
- Batch size optimization for Apple Silicon
- Memory management utilities

### 5. System Utilities (`mac_utils/system.py`)

macOS-specific system functions:

- Console management via AppleScript
- Memory information retrieval
- MPS device information
- System optimization functions

### 6. Enhanced Logging (`mac_utils/logger.py`)

Robust logging system:

- Console and file logging
- Function call decoration for tracing
- Memory usage tracking
- System information logging

## GPU Acceleration

The project now supports three GPU backends:

1. **CUDA** - For NVIDIA GPUs on Windows/Linux
2. **MPS** - For Apple Silicon GPUs on macOS
3. **CPU** - Fallback for all platforms

The system automatically detects the available backend and configures PyTorch accordingly.

## Key Modifications

1. **Platform Detection**:
   - Early detection of operating system and hardware
   - Special path for Apple Silicon Macs

2. **Console Management**:
   - Windows uses `ctypes.windll` calls
   - macOS uses AppleScript via subprocess

3. **Device Selection**:
   - CUDA is preferred when available
   - MPS is used on Apple Silicon Macs
   - CPU is the final fallback

4. **Memory Optimization**:
   - Conservative batch sizes on Apple Silicon
   - Attention slicing for large models
   - Sequential CPU offloading for memory-intensive operations

## Installation and Usage

See the [MAC_USAGE.md](MAC_USAGE.md) document for detailed instructions on installing and using Pallaidium on macOS.

## Known Limitations

1. Some diffusion models may have compatibility issues with MPS
2. Memory usage is higher on unified memory systems like Apple Silicon
3. Performance may vary between model types and sizes

## Future Improvements

1. Further memory optimizations for large models
2. Mac-specific UI adjustments
3. Better error handling for MPS-specific issues
4. Integration with Apple's ML frameworks

## Contributors

This macOS port was developed by Parsa with assistance from the Pallaidium community.

## License

This project maintains the same license as the original Pallaidium project.
