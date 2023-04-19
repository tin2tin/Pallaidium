# Generative AI - Text to Video
AI generate video and audio from text prompts. 

## How to install
Download the add-on: https://github.com/tin2tin/text_to_video/archive/refs/heads/main.zip

Run Blender as Administrator and install the add-on as usual: Preferences > Add-ons > Install > select file > Enable the add-on. 

Now, in the add-on preferences, hit the "Install all Dependencies" button, and many GB will have to be downloaded, so go grab a coffee. 

### Weights
Pruned 6 GB VRAM(included in the add-on download):

https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/tree/main

20 GB of VRAM:

https://modelscope.cn/models/damo/text-to-video-synthesis/files

Anime - 6 GB VRAM:

https://huggingface.co/kabachuha/animov-0.1-modelscope-original-format

## Location

Video Sequence Editor > Sidebar > Generative AI

![image](https://user-images.githubusercontent.com/1322593/233038942-ae01ed61-9977-4478-b90a-af8282d6556c.png)

Converting Text strips into GeneratorAI strips:

![image](https://user-images.githubusercontent.com/1322593/232625894-6726d407-c802-4619-864a-0b8b7faeceff.png)

Install Dependencies and Sound Notification in the add-on preferences:

![image](https://user-images.githubusercontent.com/1322593/233042178-8a7d300e-6093-4a95-ab79-13024e0af60e.png)

## How to
In order to make Blender run the generated video properly, the resolution and the fps should be set to match the footage. 
The resolution can be set by selecting a strip > Strip Menu > Movie Strip > Set Render Size.
The project/scene fps can be set in the Properties > Output Tab > Format > Frame Rate - set it to Custom and 8. 

Alternatively can this .blend be loaded as a quick start, but the add-on still needs to be installed:

https://github.com/tin2tin/text_to_video/raw/main/text2video_ui.blend

# Text to Audio

Find documentation here: https://github.com/haoheliu/AudioLDM
Try prompts like: Bag pipes playing a funeral dirge, punk rock band playing hardcore song, techno dj playing deep bass house music, and acid house loop with jazz.
Or: Voice of God judging mankind, woman talking about celestial beings, hammer on wood.


## Modules
Diffusers: https://github.com/huggingface/diffusers

ModelScope: https://modelscope.cn/models/damo/text-to-video-synthesis/summary

AudioLDM: https://github.com/haoheliu/AudioLDM

If some additional python modules are missing, write about it here(so I can add them), and use the Blender PIP add-on to manually install the missing modules:

https://github.com/amb/blender_pip


## Disclaimer for using the Modelscope model:

The model can only be used for non-commercial purposes. The model is meant for research purposes.

The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities.

It is prohibited to generate content that is demeaning or harmful to people or their environment, culture, religion, etc.

Prohibited for pornographic, violent and bloody content generation.

Prohibited for error and false information generation.








