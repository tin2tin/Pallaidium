"""Video/image-to-audio (and text-to-audio) via MMAudio."""

from fractions import Fraction

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename


class MMAudioPlugin(ModelPlugin):
    MODEL_ID     = "MMAudio"
    DISPLAY_NAME = "Video to Audio: MMAudio"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Generate audio for video or image clips via MMAudio"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.VIDEO
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.NEG_PROMPT,
        UISection.VIDEO_STRIP,
        UISection.AUDIO_DURATION,
        UISection.STEPS,
        UISection.GUIDANCE,
        UISection.SEED,
    ]
    PARAMS       = ParamSpec(steps=25, guidance=4.5, audio_length=8.0)
    REQUIRED_PACKAGES = ["torch", "torchaudio", "mmaudio", "librosa"]

    def load(self, prefs, scene, **kw):
        import sys
        import types

        # wandb crashes on import with NumPy 2.0 / protobuf mismatches;
        # timm only needs it for training logging, so stub it out.
        if "wandb" not in sys.modules:
            _wandb_stub = types.ModuleType("wandb")
            _wandb_stub.log = lambda *a, **k: None
            sys.modules["wandb"] = _wandb_stub

        import torch
        from mmaudio.eval_utils import ModelConfig, all_model_cfg
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        import bpy
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device = gfx_device
        dtype = torch.bfloat16

        print("Loading MMAudio…")
        model_config: ModelConfig = all_model_cfg["large_44k_v2"]
        model_config.download_if_needed()

        model = get_my_mmaudio(model_config.model_name).to(device, dtype).eval()
        model.load_weights(
            torch.load(model_config.model_path, map_location=device, weights_only=True)
        )
        print(f"MMAudio weights loaded from {model_config.model_path}")

        feature_extractor = FeaturesUtils(
            tod_vae_ckpt=model_config.vae_path,
            synchformer_ckpt=model_config.synchformer_ckpt,
            enable_conditions=True,
            mode=model_config.mode,
            bigvgan_vocoder_ckpt=model_config.bigvgan_16k_path,
            need_vae_encoder=False,
        ).to(device, dtype)

        return {
            "pipe": None,
            "model": model,
            "vocoder": model_config,   # store config for seq_cfg access
            "feature_extractor": feature_extractor,
        }

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        import torchaudio
        import bpy
        import os
        from mmaudio.eval_utils import (
            all_model_cfg, generate, load_video, load_image, make_video, VideoInfo,
        )
        from mmaudio.model.flow_matching import FlowMatching

        model           = pipe_obj["model"]
        model_config    = pipe_obj["vocoder"]   # ModelConfig stored at load time
        feature_extractor = pipe_obj["feature_extractor"]

        device = gfx_device
        seed   = inputs.seed
        scheduler = FlowMatching(
            min_sigma=0,
            inference_mode="euler",
            num_steps=inputs.steps,
        )
        scheduler_config = model_config.seq_cfg

        if torch.cuda.is_available() and seed != 0:
            generator = torch.Generator("cuda").manual_seed(seed)
        elif seed != 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None

        # ── video input ────────────────────────────────────────────────────
        if inputs.video_path and os.path.isfile(inputs.video_path):
            self.set_phase(inputs, "Generating audio from video")
            video_data = load_video(inputs.video_path, inputs.audio_length)
            video_frames = video_data.clip_frames.unsqueeze(0)
            sync_frames  = video_data.sync_frames.unsqueeze(0)
            scheduler_config.duration = video_data.duration_sec
            model.update_seq_lengths(
                scheduler_config.latent_seq_len,
                scheduler_config.clip_seq_len,
                scheduler_config.sync_seq_len,
            )
            with torch.no_grad():
                generated_audio = generate(
                    video_frames, sync_frames, [inputs.prompt],
                    negative_text=[inputs.neg_prompt],
                    feature_utils=feature_extractor,
                    net=model, fm=scheduler, rng=generator,
                    cfg_strength=inputs.guidance,
                )
            audio_output = generated_audio.float().cpu()[0]
            target_sr    = int(bpy.context.preferences.system.audio_sample_rate.split("_")[1])
            filename     = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
            make_video(video_data, filename, audio_output, sampling_rate=target_sr)
            print(f"MMAudio saved video+audio to {filename}")
            return filename

        # ── image input ────────────────────────────────────────────────────
        if scene.image_path and os.path.isfile(scene.image_path):
            self.set_phase(inputs, "Generating audio from image")
            image_data   = load_image(scene.image_path)
            clip_frames  = image_data.clip_frames.unsqueeze(0)
            sync_frames  = image_data.sync_frames.unsqueeze(0)
            fps_num = bpy.context.scene.render.fps
            fps_den = bpy.context.scene.render.fps_base or 1
            fps     = Fraction(fps_num / fps_den).limit_denominator(1001)
            video_data = VideoInfo.from_image_info(image_data, inputs.audio_length, fps=fps)
            scheduler_config.duration = inputs.audio_length
            model.update_seq_lengths(
                scheduler_config.latent_seq_len,
                scheduler_config.clip_seq_len,
                scheduler_config.sync_seq_len,
            )
            with torch.no_grad():
                generated_audio = generate(
                    clip_frames, sync_frames, [inputs.prompt],
                    negative_text=[inputs.neg_prompt],
                    feature_utils=feature_extractor,
                    net=model, fm=scheduler, rng=generator,
                    cfg_strength=inputs.guidance,
                    image_input=True,
                )
            audio_output = generated_audio.float().cpu()[0]
            target_sr    = int(bpy.context.preferences.system.audio_sample_rate.split("_")[1])
            filename     = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
            make_video(video_data, filename, audio_output, sampling_rate=target_sr)
            print(f"MMAudio saved image+audio to {filename}")
            return filename

        # ── text-only (no visual input) ────────────────────────────────────
        self.set_phase(inputs, "Generating audio from text")
        scheduler_config.duration = inputs.audio_length
        model.update_seq_lengths(
            scheduler_config.latent_seq_len,
            scheduler_config.clip_seq_len,
            scheduler_config.sync_seq_len,
        )
        with torch.no_grad():
            generation = generate(
                None, None, [inputs.prompt],
                negative_text=[inputs.neg_prompt],
                feature_utils=feature_extractor,
                net=model, fm=scheduler, rng=generator,
                cfg_strength=inputs.guidance,
                image_input=True,
            )
        audio_output = generation.float().cpu()[0]
        target_sr    = int(bpy.context.preferences.system.audio_sample_rate.split("_")[1])
        filename     = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".wav")
        torchaudio.save(filename, audio_output, target_sr)
        print(f"MMAudio text-to-audio saved to {filename}")
        return filename
