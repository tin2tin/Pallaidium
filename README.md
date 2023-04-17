# Generative AI - Text to Video
Adding text to video to the Blender Video Sequence Editor using ModelScope. The full version needs 20 GB VRAM to work. An alternative version with a pruned data set(smaller) will download as default, for now - see code to change to 20 GB VRAM version.

## How to install
Download the add-on: https://github.com/tin2tin/text_to_video/archive/refs/heads/main.zip

Run Blender as Administrator and install the add-on as usual: Preferences > Add-ons > Install > select file > Enable the add-on. When running the first generation of video or audio many GB will have to be downloaded, so go grab a coffee. 

Alternative download for the weighted set which needs 20 GB of VRAM to run:
https://modelscope.cn/models/damo/text-to-video-synthesis/files

Or a pruned weighted set which should be able to run on 6 GB VRAM:
https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/tree/main

## Location

Video Sequence Editor > Sidebar > Generative AI

![image](https://user-images.githubusercontent.com/1322593/232339521-fa87dd3d-7e54-4a18-ad4a-1d6cc5c7398c.png)

Notification sound can be changed or switched on/off under Preferences:

![image](https://user-images.githubusercontent.com/1322593/232183893-b0d4a413-d11a-4fc8-965e-172d83ff60aa.png)


## How to
In order to make Blender run the generated video properly, the resolution and the fps should be set to match the footage. 
The resolution can be set by selecting a strip > Strip Menu > Movie Strip > Set Render Size.
The project/scene fps can be set in the Properties > Output Tab > Format > Frame Rate - set it to Custom and 8. 

Alternatively can this .blend be loaded as a quick start, but the add-on still needs to be installed:

https://github.com/tin2tin/text_to_video/raw/main/text2video_ui.blend

# Text to Audio

Find documentation here: https://github.com/haoheliu/AudioLDM

Issue: https://github.com/huggingface/diffusers/issues/3091

## Modules
Diffusers: https://github.com/huggingface/diffusers

ModelScope: https://modelscope.cn/models/damo/text-to-video-synthesis/summary

AudioLDM: https://github.com/haoheliu/AudioLDM

If some additional python modules are missing, write about it here(so I can add them), and use the Blender PIP add-on to manually install the missing modules:

https://github.com/amb/blender_pip







