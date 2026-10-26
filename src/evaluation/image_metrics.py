"""Pixel-domain reconstruction-quality metrics (MSE, PSNR, SSIM).

Used by the VAE reconstruction/perturbation experiments, where — unlike the
per-dimension std comparisons used elsewhere in src/evaluation/ for the MAE
embeddings — we want to measure fidelity directly in image space, since a
well-behaved autoencoder is expected to reconstruct closely.
"""
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_array(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert('RGB'))
    return np.asarray(img)


def mse(a, b) -> float:
    a = _to_array(a).astype(np.float64)
    b = _to_array(b).astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a, b) -> float:
    a = _to_array(a)
    b = _to_array(b)
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a, b) -> float:
    a = _to_array(a)
    b = _to_array(b)
    return float(structural_similarity(a, b, data_range=255, channel_axis=-1))


def all_metrics(a, b) -> dict:
    return {'mse': mse(a, b), 'psnr': psnr(a, b), 'ssim': ssim(a, b)}
