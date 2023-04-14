# Text to Video
Adding text to video to the Blender Video Sequence Editor using ModelScope. The full version needs 20 GB VRAM to work. An alternative version with a pruned data set(smaller) will download as default, for now - see code to change to 20 GB VRAM version.

## How to install
Download the add-on: https://github.com/tin2tin/text_to_video/archive/refs/heads/main.zip

Run Blender as Administrator and install the add-on as usual: Preferences > Add-ons > Install > select file > Enable the add-on. When running the first generation of video or audio many GB will have to be downloaded, so go grab a coffee. 

Alternative download for the weighted set which needs 20 GB of VRAM to run:
https://modelscope.cn/models/damo/text-to-video-synthesis/files

Or a pruned weighted set which should be able to run on 6 GB VRAM:
https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/tree/main

## Location

Video Sequence Editor > Sidebar > Generator

![image](https://user-images.githubusercontent.com/1322593/232002471-c487eea2-fd56-4dca-a7cc-f5b43b46516f.png)

# Text to Audio
Currently, not working. Produces a file without any content.

https://github.com/huggingface/diffusers/issues/3091

## Modules
Diffusers: https://github.com/huggingface/diffusers

ModelScope: https://modelscope.cn/models/damo/text-to-video-synthesis/summary

AudioLDM: https://github.com/haoheliu/AudioLDM

If some additional python modules are missing, write about it here(so I can add them), and use the Blender PIP add-on to manually install the missing modules:

https://github.com/amb/blender_pip







