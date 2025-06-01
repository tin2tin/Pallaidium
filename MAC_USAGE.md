# Using Pallaidium on macOS

This guide provides detailed instructions for installing and using Pallaidium on macOS, with specific focus on Apple Silicon (M-series) Macs.

## System Requirements

- macOS 11 (Big Sur) or newer
- Blender 3.3 or newer
- Apple Silicon Mac (M1/M2/M3 series) for GPU acceleration
- At least 16GB RAM recommended (8GB minimum)
- 2GB+ free disk space

## Installation

### 1. Install Blender

1. Download the macOS version of Blender compatible with your Mac:
   - For Apple Silicon Macs: Use the ARM64 native version
   - For Intel Macs: Use the Intel version
   - Available at [blender.org/download](https://www.blender.org/download/)

2. Install Blender by dragging it to your Applications folder

### 2. Install Pallaidium Add-on

1. Download or clone the Pallaidium-4Mac repository
2. Open Blender
3. Go to Edit → Preferences → Add-ons → Install
4. Navigate to and select the `Pallaidium-4Mac` folder
5. Enable the add-on by checking the box next to "Pallaidium - Generative AI"

### 3. Install Dependencies

1. After enabling the add-on, go to the Pallaidium tab in the 3D Viewport sidebar (press N to show)
2. Click "Install Dependencies" in the Pallaidium panel
3. Wait for the installation to complete (this may take several minutes)
4. When prompted, restart Blender to complete the installation

## First-time Configuration

1. After restarting Blender, the Pallaidium tab should be available in the sidebar
2. Open the system console to view debugging information:
   - In Blender, go to Window → Toggle System Console, or
   - Enable "Show System Console" in the Pallaidium preferences

3. Verify GPU acceleration:
   - The console should display: "Using MPS (Metal Performance Shaders) for GPU acceleration"
   - If not, check the troubleshooting section below

## Using Pallaidium on macOS

### Optimizing for Apple Silicon

1. In the Pallaidium preferences panel, adjust these settings for best performance:
   - Reduce batch size to 1-2 for large models
   - Enable "Memory Optimization" for large generations
   - Consider reducing resolution for complex generations

2. Monitor GPU memory usage:
   - Apple Silicon has unified memory, so monitor overall system memory
   - Use Activity Monitor to check memory pressure

### Mac-specific Features

1. Console Management:
   - The macOS Terminal will be used instead of the Windows Command Prompt
   - Console visibility can be toggled in the preferences

2. Path Handling:
   - All paths are automatically normalized for macOS compatibility
   - Use forward slashes (/) for all manual path inputs

## Troubleshooting

### Common Issues

1. **"MPS not available" error**:
   - Ensure you're using macOS 12.3+ on Apple Silicon
   - Check that PyTorch 2.0+ is installed with MPS support
   - Try reinstalling dependencies

2. **Out of Memory Errors**:
   - Reduce batch size in preferences
   - Generate at lower resolutions
   - Close other memory-intensive applications
   - Enable "Memory Optimization" in preferences

3. **Slow Performance**:
   - Some models run slower on MPS than CUDA
   - Enable attention slicing in preferences
   - Use smaller models for faster generation

4. **Model Compatibility Issues**:
   - Some models may not be fully compatible with MPS
   - Try using CPU mode for problematic models
   - Check logs for specific compatibility warnings

### Accessing Logs

1. Log files are stored in the `mac_utils/logs` directory
2. Sort by date to find the most recent log file
3. Look for ERROR or WARNING messages that might explain issues

## Performance Tips

1. **Memory Management**:
   - Restart Blender periodically for long sessions
   - Use the "Clean Memory" button after large generations
   - Monitor system memory in Activity Monitor

2. **Model Selection**:
   - Smaller models (base SD 1.5) work best on MPS
   - SDXL and other large models may require memory optimization

3. **System Optimization**:
   - Keep macOS updated to the latest version
   - Close other GPU-intensive applications
   - Ensure adequate cooling for sustained performance

## Getting Help

If you encounter issues not covered in this guide:

1. Check the console and log files for specific error messages
2. Look for similar issues in the Pallaidium GitHub repository
3. Submit detailed bug reports including:
   - macOS version
   - Blender version
   - Mac model and chip (M1, M2, etc.)
   - Log files
   - Steps to reproduce the issue
