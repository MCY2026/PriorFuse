"""Frozen MedCLIP and BrainIAC encoders, followed by trainable medical prior alignment."""

import math
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale


MODALITIES = ("CTA", "dADC", "SB1000")
SLICE_COUNTS = dict(zip(MODALITIES, (5, 3, 3)))
MEDCLIP_ARCHIVE_URL = "https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip"
MEDCLIP_WEIGHT_FILE = "pytorch_model.bin"
BRAIN_IAC_SIZE = (96, 96, 96)


def ensure_medclip_weights(directory):
    """Stage the official archive and publish the checkpoint only after extraction."""
    destination = directory / MEDCLIP_WEIGHT_FILE
    if destination.is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=directory) as temp_dir:
        archive_path = Path(temp_dir) / "medclip-vit.zip"
        staged_weight = Path(temp_dir) / MEDCLIP_WEIGHT_FILE
        torch.hub.download_url_to_file(MEDCLIP_ARCHIVE_URL, str(archive_path))
        with ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if Path(name).name == MEDCLIP_WEIGHT_FILE]
            if len(members) != 1:
                raise RuntimeError(f"Expected one {MEDCLIP_WEIGHT_FILE} in the official MedCLIP archive")
            with archive.open(members[0]) as source, staged_weight.open("wb") as target:
                shutil.copyfileobj(source, target)
        staged_weight.replace(destination)


def load_medclip(directory):
    from medclip import MedCLIPVisionModelViT
    from medclip.dataset import MedCLIPFeatureExtractor
    from transformers import SwinConfig, SwinModel

    class MedCLIPImageEncoder(MedCLIPVisionModelViT):
        def __init__(self):
            # Reuse MedCLIP's vision forward with the exact Swin-T architecture.
            # The official MedCLIP checkpoint already contains the full vision branch.
            nn.Module.__init__(self)
            self.model = SwinModel(SwinConfig(
                image_size=224, patch_size=4, num_channels=3, embed_dim=96,
                depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
            ))
            self.projection_head = nn.Linear(768, 512, bias=False)

    try:
        ensure_medclip_weights(directory)
        checkpoint = torch.load(directory / MEDCLIP_WEIGHT_FILE, map_location="cpu", weights_only=True)
        vision_state = {
            name.removeprefix("vision_model."): value
            for name, value in checkpoint.items() if name.startswith("vision_model.")
        }
        model = MedCLIPImageEncoder()
        model.load_state_dict(vision_state, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Cannot load official MedCLIP ViT weights from {directory}: {exc}") from exc
    return model, MedCLIPFeatureExtractor()


def load_brainiac(weights):
    from monai.networks.nets import ViT

    if not weights.is_file():
        raise FileNotFoundError(f"BrainIAC checkpoint not found: {weights}")
    model = ViT(
        in_channels=1, img_size=BRAIN_IAC_SIZE, patch_size=(16, 16, 16),
        hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
        classification=False,
    )
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        encoder_state = {
            name.removeprefix("backbone."): value
            for name, value in state.items() if name.startswith("backbone.")
        }
        model.load_state_dict(encoder_state, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Cannot load official BrainIAC encoder from {weights}: {exc}") from exc
    return model


def brainiac_volume(slices):
    """Adapt [batch, slices, 1, H, W] to BrainIAC's single-channel 3D input."""
    # BrainIAC volumes follow NIfTI's (X, Y, Z) spatial axis order.
    axes = slices.permute(0, 2, 4, 3, 1).float()
    volume = F.interpolate(axes, size=BRAIN_IAC_SIZE, mode="trilinear", align_corners=False)
    spatial_axes = (2, 3, 4)
    foreground = volume.ne(0)
    count = foreground.sum(dim=spatial_axes, keepdim=True).clamp_min(1)
    mean = volume.sum(dim=spatial_axes, keepdim=True) / count
    centered = (volume - mean).masked_fill(~foreground, 0)
    std = (centered.square().sum(dim=spatial_axes, keepdim=True) / count).sqrt()
    return centered / std.clamp_min(1e-6)


class FrozenMedicalPriors(nn.Module):
    output_widths = {"CTA": 512, "dADC": 768, "SB1000": 768}

    def __init__(self, medclip, cta_processor, brainiac):
        super().__init__()
        self.medclip = medclip
        self.cta_processor = cta_processor
        self.brainiac = brainiac
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, medclip_dir, brainiac_weights):
        medclip, cta_processor = load_medclip(medclip_dir)
        return cls(medclip, cta_processor, load_brainiac(brainiac_weights))

    def train(self, mode=True):
        return super().train(False)

    def prepare_cta(self, cta_slices):
        """Run the standard image-only processor on the CPU before batch MixUp."""
        batch, slices = cta_slices.shape[:2]
        rgb = cta_slices.flatten(0, 1).permute(0, 2, 3, 1)
        pixels = rgb.mul(255).round().to(torch.uint8).contiguous().numpy()
        images = [Image.fromarray(image).convert("L") for image in pixels]
        result = self.cta_processor(images=images, return_tensors="pt")["pixel_values"]
        return result.reshape(batch, slices, *result.shape[1:])

    @torch.no_grad()
    def forward(self, images, cta_pixels):
        batch, slices = cta_pixels.shape[:2]
        cta_tokens = self.medclip(cta_pixels.flatten(0, 1)).reshape(batch, slices, -1)
        priors = {"CTA": cta_tokens}
        for modality in ("dADC", "SB1000"):
            # MPAM attends over BrainIAC's final patch tokens for each MRI sequence.
            grayscale = rgb_to_grayscale(images[modality])
            priors[modality], _ = self.brainiac(brainiac_volume(grayscale))
        return priors


class MedicalPriorAlignment(nn.Module):
    """Aggregate modality-specific medical features for the training loss."""

    def __init__(self, width, prior_widths):
        super().__init__()
        self.query = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width, width), nn.LeakyReLU(0.1), nn.Linear(width, width))
            for name in MODALITIES
        })
        self.project = nn.ModuleDict({
            name: nn.Linear(prior_widths[name], width, bias=False) for name in MODALITIES
        })
        self.scale = 1 / math.sqrt(width)

    def forward(self, contexts, raw_priors):
        medical_features = []
        for name in MODALITIES:
            query = self.query[name](contexts[name]).unsqueeze(1)
            values = self.project[name](raw_priors[name])
            attention = ((query @ values.transpose(1, 2)) * self.scale).softmax(dim=-1)
            medical = (attention @ values).squeeze(1)
            medical_features.append(medical)
        return torch.stack(medical_features, dim=1)


def difference_alignment_loss(shared, medical, temperature):
    """Contrast matching inter-patient differences independently per modality."""
    batch = len(shared)
    if batch < 2:
        return shared.new_zeros(())
    order = torch.randperm(batch, device=shared.device)
    partners = order.roll(1)
    with torch.autocast(shared.device.type, enabled=False):
        shared_delta = shared[order].float() - shared[partners].float()
        medical_delta = medical[order].float() - medical[partners].float()
        logits = shared_delta.transpose(0, 1) @ medical_delta.permute(1, 2, 0)
        targets = torch.arange(batch, device=shared.device).repeat(shared.size(1))
        return F.cross_entropy(logits.flatten(0, 1) / temperature, targets)
