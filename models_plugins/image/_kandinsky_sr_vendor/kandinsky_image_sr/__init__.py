"""Kandinsky ImageSR 2.0 Diffusers pipeline."""

from diffusers import ModelMixin
from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES

from kandinsky_image_sr.baked_transformer import KandinskyImageSRBakedTransformer3DModel
from kandinsky_image_sr.pipeline import KandinskyImageSROutput, KandinskyImageSRPipeline

# model_index.json names this package as the transformer's library. Before downloading a Hub
# repo, Diffusers rejects component libraries missing from LOADABLE_CLASSES, so register it
# the way `diffusers` itself is registered; loading then resolves the exported ModelMixin.
LOADABLE_CLASSES.setdefault(__name__, {"ModelMixin": ["save_pretrained", "from_pretrained"]})

__all__ = [
    "KandinskyImageSRBakedTransformer3DModel",
    "KandinskyImageSROutput",
    "KandinskyImageSRPipeline",
    "ModelMixin",
]
__version__ = "1.0.0"
