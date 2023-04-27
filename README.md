# Generative AI - Text to Video, Image and Audio in Blender
AI generate video, image and audio from text prompts or strips. 

## Features
* Text to video
* Text to audio
* Text to image
* Seed, Quality Steps, Frames, Word Power. 
* Batch operation of text strips into videos, audios or images.
* Model card selector
* Render finished notification
* Model Cards: Modelscope, Animov, Stable Diffusion 1.5 & 2, AudioLMD and Bark 

## How to install
Download the add-on: https://github.com/tin2tin/text_to_video/archive/refs/heads/main.zip

Run Blender as Administrator and install the add-on as usual: Preferences > Add-ons > Install > select file > Enable the add-on. 

In the add-on preferences, hit the "Install all Dependencies" button.

The first time a model is executed and many GB will have to be downloaded, so go grab a(lots of) coffee. 

If it says git is missing, you can find it here: https://git-scm.com/download/win


## Location

Video Sequence Editor > Sidebar > Generative AI

![image](https://user-images.githubusercontent.com/1322593/233038942-ae01ed61-9977-4478-b90a-af8282d6556c.png)
![image](https://user-images.githubusercontent.com/1322593/233155019-05c514a3-d7cc-4f4f-ba9c-7d77f957fd98.png)
![image](https://user-images.githubusercontent.com/1322593/233361486-e8fc96c3-1c3f-4077-af56-98bab66638c3.png)

Converting Text strips into GeneratorAI strips:

![image](https://user-images.githubusercontent.com/1322593/232625894-6726d407-c802-4619-864a-0b8b7faeceff.png)

Install Dependencies, set Movie Model Card and set Sound Notification in the add-on preferences:

![image](https://user-images.githubusercontent.com/1322593/233810577-961de9ec-ce3f-433b-a43c-dd4583fb518c.png)

## How to
In order to make Blender run the generated video properly, the resolution and the fps should be set to match the footage. 
The resolution can be set by selecting a strip > Strip Menu > Movie Strip > Set Render Size.
The project/scene fps can be set in the Properties > Output Tab > Format > Frame Rate - set it to Custom and 8. 

Alternatively can this .blend be loaded as a quick start, but the add-on still needs to be installed:

https://github.com/tin2tin/text_to_video/raw/main/text2video_ui.blend

## Tip
If the image of your renders breaks, then use the resolution from the Model Card in the Preferences.

# Text to Audio

### AudioLDM
Find AudioLDM documentation here: https://github.com/haoheliu/AudioLDM
Try prompts like: Bag pipes playing a funeral dirge, punk rock band playing hardcore song, techno dj playing deep bass house music, and acid house loop with jazz.
Or: Voice of God judging mankind, woman talking about celestial beings, hammer on wood.

### Bark
Find Bark documentation here: https://github.com/suno-ai/bark
* [laughter]
* [laughs]
* [sighs]
* [music]
* [gasps]
* [clears throat]
* — or ... for hesitations
* ♪ for song lyrics
* capitalization for emphasis of a word
* MAN/WOMAN: for bias towards speaker


## Modules
Diffusers: https://github.com/huggingface/diffusers

ModelScope: https://modelscope.cn/models/damo/text-to-video-synthesis/summary

AudioLDM: 

https://huggingface.co/cvssp/audioldm-s-full-v2

https://github.com/haoheliu/AudioLDM

Bark: https://github.com/suno-ai/bark

If some additional python modules are missing, write about it here(so I can add them), and use the Blender PIP add-on to manually install the missing modules:

https://github.com/amb/blender_pip


## Disclaimer for using the Modelscope model:

- The model can only be used for non-commercial purposes. The model is meant for research purposes.
- The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities.
- It is prohibited to generate content that is demeaning or harmful to people or their environment, culture, religion, etc.
- Prohibited for pornographic, violent and bloody content generation.
- Prohibited for error and false information generation.








