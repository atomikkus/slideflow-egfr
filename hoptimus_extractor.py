"""
hoptimus_extractor.py
---------------------
Custom SlideFlow feature extractor for H-optimus-0 (Bioptimus).

Architecture : vit_giant_patch14_reg4_dinov2
Feature dim  : 1536 (CLS token)
Input size   : 224 x 224
License      : Apache-2.0
HF repo      : bioptimus/H-optimus-0  (gated — requires HF token)

Usage:
    export HF_TOKEN=<your_token>
    from hoptimus_extractor import register_hoptimus
    register_hoptimus()
    extractor = build_feature_extractor("hoptimus0", tile_px=256)
"""

import os
import torch
import timm
from torchvision import transforms

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from slideflow.model.extractors import register_torch


class HOptimus0Features(TorchFeatureExtractor):
    """H-optimus-0 pathology foundation model (Bioptimus).

    ViT-Giant/14 with DINOv2-style register tokens, pretrained on large-scale
    pathology data. Returns 1536-dim CLS token embeddings.

    Requires HF_TOKEN env var with access to bioptimus/H-optimus-0.
    """

    tag = "hoptimus0"
    license = "Apache-2.0. See https://huggingface.co/bioptimus/H-optimus-0"
    citation = """
@misc{hoptimus0,
  author       = {Bioptimus},
  title        = {H-optimus-0: A foundation model for computational pathology},
  year         = {2024},
  url          = {https://huggingface.co/bioptimus/H-optimus-0}
}
"""

    # Pathology-specific normalization from model config
    MEAN = [0.707223, 0.578729, 0.703617]
    STD  = [0.211883, 0.230117, 0.177517]

    def __init__(self, tile_px=256, device="cuda", **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils
        self.device = torch_utils.get_device(device)

        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)

        self.model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.model.to(self.device)
        self.model.eval()

        self.num_features = 1536
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])
        self.preprocess_kwargs = dict(standardize=False)

    def dump_config(self):
        return self._dump_config(
            class_name="hoptimus_extractor.HOptimus0Features",
        )


def register_hoptimus():
    """Register H-optimus-0 with SlideFlow's extractor registry."""
    register_torch("hoptimus0")(HOptimus0Features)
