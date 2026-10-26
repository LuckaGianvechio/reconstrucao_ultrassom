"""Encode/decode ultrassom images using the Stable Diffusion VAE (stabilityai/sd-vae-ft-mse).

Mirrors serve/inference.py (the MAE equivalent) but for a VAE latent space:
there is no CLS-like summary token, the latent is a purely spatial 4-channel
grid, and by default we keep each image at its native resolution (only padded
to a multiple of 8, the VAE's required stride) rather than forcing a fixed
resize — the point of testing the VAE is to see how well it reconstructs
ultrasound images without discarding resolution.

A fixed-size variant (`preprocess_image_vae_fixed`) is provided for the
tabular-regression stage (phase 3), where every image needs the same latent
shape to line up as columns in a dataset.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from diffusers import AutoencoderKL


def load_vae_model(model_id: Optional[str] = None, device: str = 'cpu') -> AutoencoderKL:
    """Load the SD VAE (diffusers format) from the Hugging Face Hub.

    `model_id` defaults to the `VAE_MODEL_ID` env var, falling back to
    stabilityai/sd-vae-ft-mse (the diffusers-format sibling of
    stabilityai/sd-vae-ft-mse-original).
    """
    model_id = model_id or os.environ.get('VAE_MODEL_ID', 'stabilityai/sd-vae-ft-mse')
    vae = AutoencoderKL.from_pretrained(model_id)
    vae.eval()
    vae.to(torch.device(device))
    return vae


_TO_TENSOR = T.ToTensor()  # PIL [0,255] -> float tensor [0,1], shape (3, H, W)


def _pad_to_multiple_of(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """Pad (C, H, W) on the bottom/right so H and W are multiples of `multiple`."""
    _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


def preprocess_image_vae(img: Image.Image, pad_to_multiple_of: int = 8) -> torch.Tensor:
    """Native-resolution preprocessing: RGB, [0,1] -> [-1,1], padded to a multiple of 8."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    x = _TO_TENSOR(img)
    x = _pad_to_multiple_of(x, pad_to_multiple_of)
    x = x * 2.0 - 1.0
    return x.unsqueeze(0)


def preprocess_image_vae_fixed(img: Image.Image, size: int = 256) -> torch.Tensor:
    """Fixed-size preprocessing (resize to size x size) used for the tabular dataset stage,
    where every image must produce the same latent shape."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    resize = T.Resize((size, size))
    x = _TO_TENSOR(resize(img))
    x = x * 2.0 - 1.0
    return x.unsqueeze(0)


def encode_image_vae(pil_image: Image.Image, model: AutoencoderKL, device: str = 'cpu',
                      fixed_size: Optional[int] = None) -> np.ndarray:
    """Encode an image to its VAE latent.

    Returns a (1, 4, H', W') float32 array, already scaled by
    `model.config.scaling_factor` (so it can be decoded back directly with
    `decode_embedding_vae`).
    """
    if fixed_size is not None:
        x = preprocess_image_vae_fixed(pil_image, size=fixed_size)
    else:
        x = preprocess_image_vae(pil_image)
    x = x.to(device)
    with torch.no_grad():
        latent_dist = model.encode(x).latent_dist
        latent = latent_dist.sample() * model.config.scaling_factor
    return latent.cpu().numpy()


def decode_embedding_vae(latent: np.ndarray, model: AutoencoderKL, device: str = 'cpu',
                          crop_to: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Decode a VAE latent (as produced by `encode_image_vae`) back to a PIL image.

    `crop_to`, when given, is the original (width, height) to crop the
    decoded image back to (needed when the source was padded to a multiple
    of 8 before encoding).
    """
    device_t = torch.device(device)
    latent_t = torch.from_numpy(np.asarray(latent)).float().to(device_t)
    if latent_t.dim() == 3:
        latent_t = latent_t.unsqueeze(0)

    with torch.no_grad():
        decoded = model.decode(latent_t / model.config.scaling_factor).sample

    img = decoded[0].clamp(-1.0, 1.0)
    img = (img + 1.0) / 2.0  # [-1,1] -> [0,1]
    img = img.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)

    if crop_to is not None:
        target_w, target_h = crop_to
        img = img[:target_h, :target_w, :]

    img_u8 = (img * 255.0).round().astype('uint8')
    return Image.fromarray(img_u8)
