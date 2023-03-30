# Text to Video
Adding text to video to the Blender Video Sequence Editor using ModelScope. The full version needs 20 GB VRAM to work. An alternative version with a pruned data set(smaller) will download as default, for now - see code to change to 20 GB VRAM version.

## How to install
Run Blender as Administrator and open the system console(Blender system console processsing can be stopped with Ctrl + C) before clicking the Generate Movie button, so you can see how it is progressing the module installation. It'll take a lot of time, since it is a huge and slow download.

Alternative download for the weighted set which needs 20 GB of VRAM to run:
https://modelscope.cn/models/damo/text-to-video-synthesis/files

Or a pruned weighted set which should be able to run on 6+ GB VRAM:
https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/tree/main

The downloaded models should be placen in this folder on Windows: C:\Users\Your_user_name\AppData\Roaming\Blender Foundation\Blender\3.4\scripts\addons\text_to_video-main\model

(My RTX card isn't detected as a CUDA supported card, so I can't check if this is actually working. So, if it works for you, let me know.)

## Location

Video Sequence Editor > Sidebar > Text to Video

![image](https://user-images.githubusercontent.com/1322593/226438089-2c81fceb-6cfd-4c72-b79e-e83b97b2f8f6.png)

## Module
ModelScope: https://modelscope.cn/models/damo/text-to-video-synthesis/summary
If some additional python modules are missing, write about it here(so I can add them), and use the Blender PIP add-on to manually install the missing modules: https://github.com/amb/blender_pip


