> [!WARNING]
> WARNING: Scammers are using our free software, Pallaidium, our web content, and the name of it, as on a phishing site called: "pallaidium . com". We're NOT associated with that site! Please, help us stop them, or we'll have to delete this GitHub repository.

<h1 align="center">PALLAIDIUM</h1>

<p align="center">A free generative AI movie studio integrated into the Blender Video Editor.</p>

<p align="center">
    <a href="https://discord.gg/HMYpnPzbTm"><img src="https://img.shields.io/badge/Chat%20with%20us%20on%20Discord--blue?style=social&logo=discord" alt="Chat with us" title="Chat with us"></a>
    <a href="https://twitter.com/tintwotin"><img src="https://img.shields.io/twitter/follow/tintwotin" alt="Follow us on X" title="Follow us on X"></a>
<p>

<hr>

![PallAIdium](https://github.com/tin2tin/Generative_AI/assets/1322593/1b1b232f-00d9-4b0b-86fb-5f0f24136d2c)
AI-generate video, image, and audio from text prompts or video, image, or text strips. 


## Features

|                                                    |                                                     |                                                     |
|----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| Text to video                                      | Text to image                                       | Text to text                                        |
| Text to speech                                     | Text to audio                                       | Text to music                                       |
| Image to image                                     | Image to video                                      | Image to text                                       |
| Video to video                                     | Video to Image                                      | Video to text                                       |
| ControlNet                                         | OpenPose                                            | Canny                                               |
| ADetailer                                          | IP Adapter Face                                     | IP Adapter Style                                    |
| Multiple LoRAs                                     | LoRA Weight                                         | Style selector                                      |
| Seed                                               | Quality steps                                       | Strip power                                         |
| Frames (Duration)                                  | Word power                                          | Model card selector                                 |
| Batch conversion                                   | Batch refinement of images.                         | Prompt batching                                     |
| Batch upscale & refinement of movies.              | Render-to-path selector.                            | Render finished notification.                       |
| User-defined file path for generated files.        | Seed and prompt added to strip name.                | One-click install and uninstall dependencies.       |

## Requirements
* Windows (Unsupported: Linux and MacOS).
* A CUDA-supported Nvidia card with at least 6 GB VRAM.
* CUDA: 12.4
* 20+ GB HDD. (Each model is 6+ GB).

For Mac and Linux, we'll have to rely on contributor support. So, post your issues here for Mac: https://github.com/tin2tin/Pallaidium/issues/106 and here for Linux: https://github.com/tin2tin/Pallaidium/issues/105, and hope some contributor wants to help you out.

## How to install

* First, download and install git (must be on PATH): https://git-scm.com/downloads

* Download the add-on: [https://github.com/tin2tin/text_to_video/archive/refs/heads/main.zip](https://github.com/tin2tin/Pallaidium/archive/refs/heads/main.zip)

* On Windows, right-click on the Blender icon and "Run Blender as Administrator"(or you'll get write permission errors).

* Install the add-on as usual: Preferences > Add-ons > Install > select file > enable the add-on. 

* In the Generative AI add-on preferences, hit the "Uninstall Dependencies" button (to clear out any incompatible libs). 

* Restart Blender via "Run as Administrator".

* In the Generative AI add-on preferences, hit the "Install Dependencies" button.
  
* Restart the computer and run Blender via "Run as Administrator".

* Open the add-on UI in the Sequencer > Sidebar > Generative AI.

* 5-10 GB must be downloaded first the first time any model is executed. 

If any Python modules are missing, use this add-on to install them manually:     |
:------------- |
https://github.com/tin2tin/blender_pip      |
If "WARNING: Failed to find MSVC", install "Tools for Visual Studio":
https://aka.ms/vs/17/release/vs_BuildTools.exe


## Uninstall

Hugging Face Diffusers models are downloaded from the hub and saved to a local cache directory. Delete the folder manually:

On Linux and macOS: ~/.cache/huggingface/hub

On Windows: %userprofile%\\.cache\huggingface\hub

![image](https://github.com/user-attachments/assets/a5c44a7e-c670-49ef-941f-86e521568637)

## Change Log
* 2025-2-25: Add: MMAudio for Video to Sync Audio 
* 2025-2-21: Support for Skywork/SkyReels-V1-Hunyuan-T2V/I2V. Need a full update of dependencies! (Thx newgenai79 for int4 transformer)
* 2025-2-15: Add: LoRA support for HunyuanVideo + better preset  
* 2025-2-12: Add multi-media prompting via: [OmniGen](https://github.com/VectorSpaceLab/OmniGen)
* 2025-2-10: 
Update: a-r-r-o-w/LTX-Video-0.9.1-diffusers ZhengPeng7/BiRefNet_HR MiaoshouAI/Florence-2-large-PromptGen-v2.0
New: ostris/Flex.1-alpha Alpha-VLLM/Lumina-Image-2.0 Efficient-Large-Model/Sana_1600M_1024px_diffusers
Fix: Frame by frame (SD XL)
Remove: Corcelio/mobius
* 2025-1-26: Add: MiniMax Cloud txt/img/subject to video (insert your MiniMax API key in MiniMax_API.txt) and fast FLUX LoRA
* 2025-1-15: FLUX: faster img2img and inpaint
* 2024-11-2: Add: Image Background Removal, Stable Diffusion 3.5 Medium, Fast Flux(t2i)
* 2024-9-19: Add: Image to Video for CogVideoX
* 2024-9-15: Add: LoRA import for Flux
* 2024-9-14: Add: Flux Inpaint & Img2img.
* 2024-9-4: Add: Florence 2 (Image Caption), AudioLDM2-Large, CogVideox-2b, flash_attn on Win.
* 2024-9-2: Add: Vid2vid for CogVideoX-5b and Parler TTS
* 2024-8-28: Make CogVideox-5b run on 6 GB VRAM & Flux on 2 GB VRAM
* 2024-8-27: Add: CogVideoX-5b Remove: Low-quality models
* 2024-8-5: Add: Flux Dev - NB. needs update of dependencies and 24 GB VRAM
* 2024-8-2: Add: Flux Schnell - NB. needs update of dependencies and 24 GB VRAM
* 2024-7-12: Add: Kwai/Kolors (txt2img & img2img)
* 2024-6-13: Add: SD3 - A "Read" token from HuggingFace must be entered, it's free (img2img). Fix: Installation of Dependencies
* 2024-6-6: Add: Stable Audio Open, Frame:-1 will inherit duration. 
* 2024-6-1: IP Adapter(When using SDXL): Face (Image or folder), Style (image or folder) New image models: Mobius, OpenVision, Juggernaut X Hyper
* 2024-4-29: Add: PixArt Sigma 2k, PixArt 1024 and RealViz V4
* 2024-2-23: Add: Proteus Lightning and Dreamshaper XL Lightning
* 2024-2-21: Add: SDXL-Lightning 2 Step & Proteus v. 0.3
* 2024-1-02: Add: WhisperSpeech
* 2024-01-01: Fix installation and Bark bugs.
* 2024-01-31: Add OpenDalle, Speed option, SDXL, and LoRA support for Canny and OpenPose, including OpenPose rig images. Prune old models including SD. 
* 2023-12-18: Add: Bark audio enhance, Segmind Vega. 
* 2023-12-1: Add SD Turbo & MusicGen Medium, MPS device for MacOS.
* 2023-11-30: Add: SVD, SVD-XT, SDXL Turbo

## Location

Install Dependencies, and set Sound Notification in the add-on preferences:

![image](https://github.com/tin2tin/Generative_AI/assets/1322593/49ba0182-f8a0-4a1d-b24f-caca9741d033)

Video Sequence Editor > Sidebar > Generative AI:

![image](https://github.com/tin2tin/Pallaidium/assets/1322593/e3c1193d-5e0a-4ed2-acca-3f7a4413e4c1)

## Styles:

![image](https://github.com/tin2tin/Generative_AI/assets/1322593/86807264-a377-4de1-875e-471aaa3011a7)

See SDXL handling most of the styles here: https://stable-diffusion-art.com/sdxl-styles/

## Tips:
- If the image of your renders breaks, use the resolution from the Model Card in the Preferences.
- If the image of your playback stutters, then select a strip > Menu > Strip > Movie Strip > Set Render Size.
- If you get the message that CUDA is out of memory, restart Blender to free up memory and make it stable again.
- New to Blender? Watch this tutorial: https://youtu.be/4_MIaxzjh5Y?feature=shared

## Batch Processing

Select multiple strips and hit Generate. When doing this, the file name, and if found the seed value, are automatically inserted into the prompt and seed value. However, in the add-on preferences, this behavior can be switched off.

https://github.com/tin2tin/Pallaidium/assets/1322593/28098eb6-3a93-4bcb-bd6f-53b71faabd8d

## Text to Audio

### Voices for Parler TTS: 
Laura, Gary, Jon, Lea, Karen, Rick, Brenda, David, Eileen, Jordan, Mike, Yann, Joy, James, Eric, Lauren, Rose, Will, Jason, Aaron, Naomie, Alisa, Patrick, Jerry, Tina, Jenna, Bill, Tom, Carol, Barbara, Rebecca, Anna, Bruce, Emily

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
* capitalization for emphasis on a word
* MAN/WOMAN: for bias towards the speaker

Speaker Library: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c

If the audio breaks up           |
:------------- |
Try processing longer sentences.      |

## Useful add-ons

### GPT4BLENDER

Use GPT4ALL to generate image prompts or stories:

https://github.com/tin2tin/GPT4BLENDER

![image](https://github.com/user-attachments/assets/464e0fe3-0994-4920-9ceb-ef68b331866b)

### Text to Strip

Convert text from the Text Editor to strips which can be used as prompts for batch generation.

https://github.com/tin2tin/text_to_strip

![image](https://github.com/user-attachments/assets/f829d338-31ba-45fc-845c-eb563e14ea77)


### Subtitle Editor

Edit, navigate, and i/o text strips.

https://github.com/tin2tin/Subtitle_Editor

![image](https://github.com/user-attachments/assets/6e565c39-b77a-4c13-9726-39fb9f73db58)

### VSE Masking Tools

For creating a mask on top of a clip in the Sequencer, this add-on can be used to input the clip as background in the Blender Image Editor. The created mask can then be added to the VSE as a strip, and converted to video with the above add-on:

https://github.com/tin2tin/vse_masking_tools

![image](https://github.com/tin2tin/Generative_AI/assets/1322593/f2afd36c-34b1-4779-957b-0eb8defed296)

### Add Rendered Strips

Since the Generative AI add-on can only input images or movie strips, you'll need to convert other strip types to movie-strip. For this purpose, this add-on can be used:

https://github.com/tin2tin/Add_Rendered_Strips

![image](https://github.com/tin2tin/Generative_AI/assets/1322593/d8c0a184-d812-440d-a5a8-501a1282d78d)

### Low on VRAM?
Disable System memory fallback: https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion

## Useful Projects

### Trainer for LoRAs: 
https://github.com/Nerogar/OneTrainer


## Video Examples

### VID2VID & TXT2VID
[![Watch the video](https://img.youtube.com/vi/S2b7QGv-l-o/0.jpg)](https://www.youtube.com/watch?v=S2b7QGv-l-o)

### Image to Text
https://github.com/tin2tin/Pallaidium/assets/1322593/91eb17e4-72d6-4c69-8e5c-a3d38af5a770


### Illusion Diffusion
https://github.com/tin2tin/Pallaidium/assets/1322593/42eadfd8-3ebf-4747-b8e0-7b79fe8626b6


### Scribble
https://github.com/tin2tin/Pallaidium/assets/1322593/c74a4e38-8b16-423b-be78-aadfbfe284dc


### Styles
https://github.com/tin2tin/Pallaidium/assets/1322593/b80812b4-e3be-40b0-a73b-bc55b7eeadf7


### Canny
https://github.com/tin2tin/Pallaidium/assets/1322593/a1e94e09-0147-40ae-b4c2-4ce0671b1289


### OpenPose
https://github.com/tin2tin/Pallaidium/assets/1322593/ac9f278e-9fc9-46fc-a4e7-562ff041964f


### Screenplay to Film
[![Watch the video](https://img.youtube.com/vi/J64ZitsSN6k/0.jpg)](https://youtu.be/J64ZitsSN6k) 


### Img to Txt to Audio
[![Watch the video](https://img.youtube.com/vi/0EnUq1RhJ6M/0.jpg)](https://youtu.be/0EnUq1RhJ6M) 


### Zeroscope
[![Watch the video](https://img.youtube.com/vi/LejSJGmtEvE/0.jpg)](https://youtu.be/LejSJGmtEvE) 


### Würstchen
[![Watch the video](https://img.youtube.com/vi/CDPmGs_JtSM/0.jpg)](https://youtu.be/CDPmGs_JtSM) 


### Bark
[![Watch the video](https://img.youtube.com/vi/AAdQfQjENJU/0.jpg)](https://youtu.be/AAdQfQjENJU) 


### Batch from Text Strips
[![Watch the video](https://img.youtube.com/vi/gSFWGkgaNsE/0.jpg)](https://youtu.be/gSFWGkgaNsE)


### Video to video:
https://github.com/tin2tin/Generative_AI/assets/1322593/c044a0b0-95c2-4b54-af0b-45bc0c670c89

https://github.com/tin2tin/Generative_AI/assets/1322593/0105cd35-b3b2-49cf-91c1-0633dd484177

### Img2img:
https://github.com/tin2tin/Generative_AI/assets/1322593/2dd2d2f1-a1f6-4562-8116-ffce872b79c3

### Painting
https://github.com/tin2tin/Generative_AI/assets/1322593/7cd69cd0-5842-40f0-b41f-455c77443535

## Restrictions on using Pallaidium:
- It is prohibited to use Pallaidium to generate content that is demeaning or harmful to people, their environment, culture, religion, etc.
- It is prohibited to use Pallaidium for pornographic, violent, and bloody content generation.
- It is prohibited to use Pallaidium for error and false information generation.

## Restrictions on using the AI models:
- I, tintwotin, do not endorse or take responsibility for third-party use.
- Pallaidium does not include any genAI models(weights). If the user decides to use a model, it is downloaded from HuggingFace. 
- In general, the models can only be used for non-commercial purposes and are meant for research purposes.
- Consult the individual models on HuggingFace to read up on their licenses and ex. if they can be used commercially.

## Credits 

- The [Diffusers](https://github.com/huggingface/diffusers) lib makes the following weights accessible through the Pallaidium UI:

### Video:
- [SkyReels-V1-Hunyuan-I2V/T2V](https://huggingface.co/Skywork/SkyReels-V1-Hunyuan-I2V)
- [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)
- [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
- [THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
- [cerspense/zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL)

### Image:
- [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)
- [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/ChuckMcSneed/FLUX.1-dev)
- [fluently/Fluently-XL-Final](https://huggingface.co/fluently/Fluently-XL-Final)
- [shuttleai/shuttle-jaguar](https://huggingface.co/shuttleai/shuttle-jaguar)
- [Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)
- [Kwai-Kolors/Kolors-diffusers](https://huggingface.co/Kwai-Kolors/Kolors-diffusers)
- [dataautogpt3/OpenDalleV1.1](https://huggingface.co/dataautogpt3/OpenDalleV1.1)
- [PixArt-alpha/PixArt-Sigma_16bit](https://huggingface.co/Vargol/PixArt-Sigma_16bit)
- [PixArt-alpha/PixArt-Sigma_2k_16bit](https://huggingface.co/Vargol/PixArt-Sigma_2k_16bit)
- [dataautogpt3/ProteusV0.4](https://huggingface.co/Vargol/ProteusV0.4)
- [SG161222/RealVisXL_V4.0](https://huggingface.co/SG161222/RealVisXL_V4.0)
- [Salesforce/blipdiffusion](https://huggingface.co/Salesforce/blipdiffusion)
- [diffusers/controlnet-canny-sdxl-1.0-small](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small)
- [xinsir/controlnet-openpose-sdxl-1.0](https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0)
- [xinsir/controlnet-scribble-sdxl-1.0](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0)
- [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
- [Flex](https://huggingface.co/ostris/Flex.1-alpha)
- [Lumina 2](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0)
- [Sana](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers)
- [OmniGen](https://github.com/VectorSpaceLab/OmniGen)

### Audio:
- [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Parler TTS](https://github.com/huggingface/parler-tts)
- [facebook/musicgen-stereo-medium](https://huggingface.co/facebook/musicgen-stereo-medium)
- [vtrungnhan9/audioldm2-music-zac2023](https://huggingface.co/vtrungnhan9/audioldm2-music-zac2023)
- [Bark](https://github.com/suno-ai/bark)
- [WhisperSpeech](https://github.com/collabora/WhisperSpeech)
- [parler-tts/parler-tts-large-v1](https://huggingface.co/parler-tts/parler-tts-large-v1)
- [MMAudio](https://github.com/hkchengrex/MMAudio)

### Background Removal:
- [BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR)

### Text Captions:
- [Florence-2-large-PromptGen](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v2.0)











