"""
dinov2_extractor.py
-------------------
Custom SlideFlow feature extractor for DINOv2 ViT-L/14 (Meta AI).

Architecture : ViT-L/14
Feature dim  : 1024 (CLS token)
Input size   : 224 x 224
Weights      : https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
License      : Apache-2.0

Usage:
    from dinov2_extractor import register_dinov2_vitl
    register_dinov2_vitl()
    extractor = build_feature_extractor("dinov2_vitl14", tile_px=256)
"""

import torch
from torchvision import transforms
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from slideflow.model.extractors._registry import register_torch


WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"

# ImageNet normalization (DINOv2 uses standard ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class DinoV2ViTL14Features(TorchFeatureExtractor):
    """DINOv2 ViT-L/14 feature extractor (Meta AI).

    Loads weights directly from Facebook CDN — no token required.
    Returns 1024-dim CLS token embeddings.
    """

    tag = "dinov2_vitl14"
    license = "Apache-2.0. See https://github.com/facebookresearch/dinov2"

    def __init__(self, tile_px=256, device="cuda", **kwargs):
        super().__init__(**kwargs)

        from slideflow.model import torch_utils
        self.device = torch_utils.get_device(device)

        print("Loading DINOv2 ViT-L/14 from torch.hub …")
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14",
            pretrained=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.num_features = 1024
        self.transform = self.build_transform(
            img_size=224,
            resize=True,       # resize 256→224 before passing to ViT-L/14
            norm_mean=MEAN,
            norm_std=STD,
        )
        self.preprocess_kwargs = dict(standardize=False)

    def dump_config(self):
        return self._dump_config(
            class_name="dinov2_extractor.DinoV2ViTL14Features",
        )


def register_dinov2_vitl():
    """Register DINOv2 ViT-L/14 with SlideFlow's extractor registry."""
    register_torch("dinov2_vitl14")(DinoV2ViTL14Features)
