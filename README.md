<h1 align="center">PALLAIDIUM</h1>

<p align="center">A free generative AI movie studio integrated into the Blender Video Editor.</p>

<p align="center">
    <a href="https://discord.gg/HMYpnPzbTm"><img src="https://img.shields.io/badge/Chat%20with%20us%20on%20Discord--blue?style=social&logo=discord" alt="Chat with us" title="Chat with us"></a>
    <a href="https://twitter.com/tintwotin"><img src="https://img.shields.io/twitter/follow/tintwotin" alt="Follow us on X" title="Follow us on X"></a>
<p>

<hr>

## Workflow examples with Pallaidium, Blender Screenwriter, and GPT4Blender

https://github.com/user-attachments/assets/81d30bc1-01f6-4b52-8ce7-abf53d53e854

AI-generate video, image, and audio from text prompts or video, image, or text strips. 

## Generation Matrix
This matrix provides a quick overview of the core generative capabilities, mapping input types to possible outputs.

| Input | Image | Video | Text | Audio | Music | Speech |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Image**| ✅ | ✅ | ✅ | ✅ | | |
| **Video**| ✅ | ✅ | ✅ | ✅ | | |

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
* [Windows](https://github.com/tin2tin/Pallaidium/wiki/Windows-Model-Status). Limited support for Linux and [MacOS](https://github.com/tin2tin/Pallaidium/wiki/macOS-Compatibility-Status)
* Blender 4.5+ https://builder.blender.org/download/daily/
* A CUDA-supported Nvidia card with at least 6 GB VRAM or MPS.
* CUDA: 12.4
* 20+ GB HDD. (Each model is 6+ GB).

For Mac and Linux, we'll have to rely on contributor support. So, post your issues here for Mac: https://github.com/tin2tin/Pallaidium/issues/106 and here for Linux: https://github.com/tin2tin/Pallaidium/issues/105, and hope some contributor wants to help you out.

## How to install

* First, download and install git (must be on PATH): https://git-scm.com/downloads

* Download Blender 4.5.3 (not 5.0) https://builder.blender.org/download/daily/ and unzip it into the Documents folder.  

* Download the add-on: [https://github.com/tin2tin/text_to_video/archive/refs/heads/main.zip](https://github.com/tin2tin/Pallaidium/archive/refs/heads/main.zip)

* On Windows, right-click on the Blender(blender.exe) icon and "Run Blender as Administrator"(or you'll get write permission errors).

* Install the add-on as usual: Preferences > Add-ons > Install > select file > enable the add-on. 

* In the Generative AI add-on preferences, hit the "Uninstall Dependencies" button (to clear out any incompatible libs). 

* Restart Blender via "Run as Administrator".

* In the Generative AI add-on preferences, hit the "Install Dependencies" button.
  
* Restart the computer and run Blender via "Run as Administrator".

* Open the add-on UI in the Sequencer > Sidebar > Generative AI.

* 5-10 GB must be downloaded first the first time any model is executed.

* When you have Pallaidium installed, reach out on Discord: https://discord.gg/HMYpnPzbTm or leave a note on how it is working for you. It means the world to me to know someone is using it! 

If any Python modules are missing, use this add-on to install them manually:     |
:------------- |
https://github.com/tin2tin/blender_pip      |
If "WARNING: Failed to find MSVC", install "Tools for Visual Studio":
https://aka.ms/vs/17/release/vs_BuildTools.exe
If error: "Missing DLL", install Microsoft Visual C++ Redistributable:
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170


## Uninstall

Hugging Face Diffusers models are downloaded from the hub and saved to a local cache directory. Delete the folder manually:

On Linux and macOS: ~/.cache/huggingface/hub

On Windows: %userprofile%\\.cache\huggingface\hub


## Usage

The addon panel is located in the **Video Sequence Editor > Sidebar (N-Panel) > Generative AI** tab.

### Basic Workflow

1.  **Choose Output:** In the "Output" section of the panel, select whether you want to generate an `Image`, `Video`, `Audio`, or `Text`.
2.  **Select Model:** Based on your output choice, select a specific AI model from the dropdown list.
3.  **Set Input:**
    -   **For Prompts:** Leave the `Input` dropdown on `Prompts`.
    -   **For Strips:** Select one or more strips in the VSE timeline and set the `Input` dropdown to `Strips`.
4.  **Configure:** Enter your text prompts and adjust parameters like resolution, quality steps, and guidance scale.
5.  **Generate:**
    -   If using `Prompts`, click the **`Generate`** button.
    -   If using `Strips`, click the **`Generate from Strips`** button.

The generated media will be saved to the directory specified in the addon preferences and automatically added to your VSE timeline on a new channel.


## Change Log
* 2025-10-1: Fix: Deps. Add: Qwen Multi-image Edit.
* 2025-7-05: Add: FLUX Kontext Relight.
* 2025-6-26: Add: FLUX.1 Dev Kontext. Update Diffusers by installing this: "git+https://github.com/huggingface/diffusers.git" with the Python Module Manager add-on (link below), and restart Blender. 
* 2025-6-22: Add: Long string parsing for Chatterbox (for Audiobooks). Use Blender 5.0 Alpha.
* 2025-6-14: Add: Chroma txt2vid
* 2025-6-12: Add: ZuluVision/MoviiGen1.1_Prompt_Rewriter
* 2025-6-01: Add: Chatterbox with zero shot text to speech and speech to speech voice cloning.
* 2025_5_29: Add: Flux depth+canny+redux Fix: neg Flux input + py libs
* 2025-5-12: Add: F5-TTS Voice Cloning
* 2025-5-09: Update: LTX 0.9.7 w. img, txt & vid input
* 2025-5-08: FramePack (Img/+Last Img), Img+txt for MMAudio, Flux De-distilled
* 2025-4-11: Improved LTX 0.95, MetaData, StripPicker, Wan t2i
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

![image](https://github.com/user-attachments/assets/44852946-67db-4788-aed0-6e383070c2ac)

See SDXL handling most of the styles here: https://stable-diffusion-art.com/sdxl-styles/

## Tips:
- If the image of your renders breaks, use the resolution from the Model Card in the Preferences.
- If the image of your playback stutters, then select a strip > Menu > Strip > Movie Strip > Set Render Size.
- If you get the message that CUDA is out of memory, restart Blender to free up memory and make it stable again.
- New to Blender? Watch this tutorial: https://youtu.be/4_MIaxzjh5Y?feature=shared

![image](https://github.com/user-attachments/assets/a5c44a7e-c670-49ef-941f-86e521568637)

## Batch Processing

Select multiple strips and hit Generate. When doing this, the file name, and if found, also the seed value, are automatically inserted into the prompt and seed value. However, in the add-on preferences, this behavior can be switched off.

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

Convert text from the Text Editor to strips, which can be used as prompts for batch generation.

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

### Pallaidium Module Checker

Add-on to check if the gen AI models are running error-free in Pallaidium. 

https://github.com/tin2tin/pallaidium_module_checker

<img width="385" height="446" alt="462758956-7dec5ead-c6d0-4846-a1ef-03c3ee47ecd0" src="https://github.com/user-attachments/assets/069dc2ad-9a25-47a1-995a-82e047d872cc" />

### Blender Screenwriter

Write screenplays, add image prompts, and convert everything to times text strips, ready for batch convering to ex. imgage, video or speech.

https://github.com/tin2tin/Blender_Screenwriter

![image](https://github.com/tin2tin/Blender_Screenwriter/blob/master/bsw_tut.gif)

### Low on VRAM?
Disable System memory fallback: https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion

## Useful Projects

### Trainer for LoRAs: 
https://github.com/Nerogar/OneTrainer


## Video Examples


### Image to Text
https://github.com/tin2tin/Pallaidium/assets/1322593/91eb17e4-72d6-4c69-8e5c-a3d38af5a770


### Scribble
https://github.com/tin2tin/Pallaidium/assets/1322593/c74a4e38-8b16-423b-be78-aadfbfe284dc


### Styles
https://github.com/tin2tin/Pallaidium/assets/1322593/b80812b4-e3be-40b0-a73b-bc55b7eeadf7


### Canny
https://github.com/tin2tin/Pallaidium/assets/1322593/a1e94e09-0147-40ae-b4c2-4ce0671b1289


### OpenPose
https://github.com/tin2tin/Pallaidium/assets/1322593/ac9f278e-9fc9-46fc-a4e7-562ff041964f


### Video to video:
https://github.com/tin2tin/Generative_AI/assets/1322593/c044a0b0-95c2-4b54-af0b-45bc0c670c89


https://github.com/tin2tin/Generative_AI/assets/1322593/0105cd35-b3b2-49cf-91c1-0633dd484177


### Frame by Frame:
https://github.com/tin2tin/Generative_AI/assets/1322593/2dd2d2f1-a1f6-4562-8116-ffce872b79c3


## Restrictions on using Pallaidium:
- The team behind Pallaidium does not endorse or take responsibility for third-party use.
- The team behind Pallaidium requires verification or explicit permission for redistribution.
- It is prohibited to use Pallaidium to generate content that is demeaning or harmful to people, their environment, culture, religion, etc.
- It is prohibited to use Pallaidium for pornographic, violent, and bloody content generation.
- It is prohibited to use Pallaidium for error and false information generation.
- It is prohibited to use Pallaidium for commercial misuse or misrepresentation.
  
## Restrictions on using the AI models:
- Pallaidium does not include any genAI models(weights). If the user decides to use a model, it is downloaded from HuggingFace. 
- In general, the models can only be used for non-commercial purposes and are meant for research purposes.
- Consult the individual models on HuggingFace to read up on their licenses and ex. if they can be used commercially.

## Credits 

- The [Diffusers](https://github.com/huggingface/diffusers) lib makes the following weights accessible through the Pallaidium UI:

### Video:
- [FramePack](https://github.com/lllyasviel/FramePack)
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
- [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
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
- [Chroma](https://huggingface.co/lodestones/Chroma)
- [Relight](https://huggingface.co/kontext-community/relighting-kontext-dev-lora-v3)

### Audio:
- [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [facebook/musicgen-stereo-medium](https://huggingface.co/facebook/musicgen-stereo-medium)
- [F5-TTS](https://github.com/SWivid/F5-TTS)
- [WhisperSpeech](https://github.com/collabora/WhisperSpeech)
- [MMAudio](https://github.com/hkchengrex/MMAudio)
- [Chatterbox](https://github.com/resemble-ai/chatterbox)

### Background Removal:
- [BiRefNet_HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR)

### Generate Text Captions/Enhancing:
- [Florence-2-large-PromptGen](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v2.0)
- [MoviiGen1.1_Prompt_Rewriter](https://huggingface.co/ZuluVision/MoviiGen1.1_Prompt_Rewriter)

### MacOS:
- [MFLUX](https://github.com/filipstrand/mflux)

> [!WARNING]
> SCAM ALERT!
> Scammers are misusing our free software, Pallaidium, along with our content and name, on a phishing site: pallaidium . com. We are NOT associated with this site!
> 🚨 Please help us report this scam — otherwise, we may be forced to delete this GitHub repository.

![PallAIdium](https://github.com/tin2tin/Generative_AI/assets/1322593/1b1b232f-00d9-4b0b-86fb-5f0f24136d2c)











