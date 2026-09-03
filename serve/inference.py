import os
import io
import sys
import base64
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


def _ensure_mae_in_path(mae_path: str = None):
    if mae_path is None:
        mae_path = os.path.join(os.path.dirname(__file__), '..', 'mae')
    mae_path = os.path.abspath(mae_path)
    if mae_path not in sys.path:
        sys.path.insert(0, mae_path)


def load_model(checkpoint_path: str, device: str = 'cpu', mae_repo_path: str = None):
    """Load MAE model from facebookresearch/mae repo and a checkpoint file.

    Returns the loaded model on the requested device. Assumes the mae repo
    is available at `mae_repo_path` (or ../mae by default).
    """
    _ensure_mae_in_path(mae_repo_path)
    try:
        from mae.models_mae import mae_vit_base_patch16_dec512d8b
    except Exception:
        # Try common alternative import
        from models_mae import mae_vit_base_patch16_dec512d8b

    device = torch.device(device)
    model = mae_vit_base_patch16_dec512d8b()
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location='cpu')
        # some checkpoints store under ['model']
        if isinstance(state, dict) and 'model' in state:
            state_dict = state['model']
        else:
            state_dict = state

        # strip potential "module." prefixes
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state[new_k] = v
        model.load_state_dict(new_state, strict=False)

    model.to(device)
    return model


# Pre/post processing utilities
_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return _PREPROCESS(img).unsqueeze(0)


def encode_image(pil_image: Image.Image, model, device: str = 'cpu', pooled: bool = True):
    """Encode image to embedding.

    If `pooled=True` returns a single pooled vector (CLS or mean-pooled),
    otherwise returns per-patch embeddings (num_patches x dim).
    """
    x = preprocess_image(pil_image).to(device)
    with torch.no_grad():
        # many MAE implementations expose `forward_encoder` or `get_intermediate_layers`
        # try common attribute names
        if hasattr(model, 'forward_encoder'):
            # request no masking to obtain full per-patch embeddings
            enc = model.forward_encoder(x, mask_ratio=0.0)
        else:
            # fallback: run model.patch_embed + encoder
            try:
                x = model.patch_embed(x)
                cls_token = model.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                enc = model.encoder(x)
            except Exception:
                # last resort: call model(x) and hope it returns a dict
                out = model(x)
                if isinstance(out, dict) and 'encoder' in out:
                    enc = out['encoder']
                else:
                    # try returning pooled output
                    if hasattr(model, 'encode'):
                        enc = model.encode(x)
                    else:
                        raise RuntimeError('Could not obtain encoder output from model')

        # enc: (B, N, D) or (B, D)
        # some implementations return (enc, mask, ids_restore)
        if isinstance(enc, (tuple, list)):
            # enc may be (x_masked, mask, ids_restore)
            x_only = enc[0]
            mask = enc[1] if len(enc) > 1 else None
            ids_restore = enc[2] if len(enc) > 2 else None
            enc = x_only
        else:
            mask = None
            ids_restore = None

        if enc.dim() == 3 and pooled:
            # mean pool over patches (skip cls token if present)
            if enc.shape[1] > 1:
                emb = enc[:, 1:, :].mean(dim=1)
            else:
                emb = enc.mean(dim=1)
        else:
            emb = enc

        emb = emb.cpu().numpy()
        # return ids_restore only when returning per-patch embeddings (pooled=False)
        if not pooled:
            if ids_restore is not None:
                return emb, ids_restore.cpu().numpy()
            return emb
        else:
            return emb


def decode_embedding(embedding: np.ndarray, model, device: str = 'cpu', ids_restore=None) -> Image.Image:
    """Decode a per-patch embedding to an image using model.decoder.

    NOTE: This function expects `embedding` to be the per-patch token embeddings
    (shape: [num_patches+1, dim] or [1, num_patches+1, dim]). If you only have
    a pooled vector, you must provide a mapping to patch embeddings first.
    """
    device = torch.device(device)
    emb_np = np.asarray(embedding)
    with torch.no_grad():
        # Prepare token sequence to match serve/test_decoder.py behavior
        num_patches = model.patch_embed.num_patches

        # Case 1: pooled vector (1,D) or (D,) -> repeat to length num_patches+1
        if emb_np.ndim == 1 or (emb_np.ndim == 2 and emb_np.shape[0] == 1 and emb_np.ndim == 2 and emb_np.shape[1] > 1 and emb_np.shape[0] == 1 and emb_np.shape[1] > 1):
            emb_t = torch.from_numpy(emb_np).float().to(device)
            if emb_t.dim() == 1:
                emb_t = emb_t.unsqueeze(0)
            x_tokens = emb_t.unsqueeze(1).repeat(1, num_patches + 1, 1)
        else:
            # per-patch embeddings: ensure shape (1, N, D)
            if emb_np.ndim == 2:
                emb_t = torch.from_numpy(emb_np).float().unsqueeze(0).to(device)
            else:
                emb_t = torch.from_numpy(emb_np).float().to(device)
            # if tokens do not include cls token (N == num_patches), prepend pooled cls as mean
            if emb_t.shape[1] == num_patches:
                cls_tok = emb_t.mean(dim=1, keepdim=True)
                x_tokens = torch.cat([cls_tok, emb_t], dim=1)
            else:
                x_tokens = emb_t

        # ids_restore
        ids_t = None
        if ids_restore is not None:
            ids_t = torch.from_numpy(np.asarray(ids_restore)).to(device)
        else:
            ids_t = torch.arange(num_patches).unsqueeze(0).to(device)

        # run decoder (prefer forward_decoder)
        try:
            recon_patches = model.forward_decoder(x_tokens, ids_t)
        except Exception:
            try:
                recon_patches = model.decoder(x_tokens)
            except Exception:
                recon_patches = None

        if recon_patches is None:
            raise RuntimeError('Model decoder unavailable or returned unexpected format')

        # reconstruct image
        try:
            imgs = model.unpatchify(recon_patches)
        except Exception:
            imgs = recon_patches

        imgs = imgs[0].cpu().numpy()  # (3, H, W)
        img = imgs.transpose(1, 2, 0)

        # If embedding appears to be per-patch (ids_restore present or emb length == num_patches+1),
        # use inverse ImageNet normalization (this produced the previously expected colors).
        use_inv_imagenet = False
        if ids_restore is not None:
            use_inv_imagenet = True
        elif emb_np.ndim == 3 and emb_np.shape[1] in (num_patches, num_patches + 1):
            use_inv_imagenet = True

        if use_inv_imagenet:
            mean = np.array([0.485, 0.456, 0.406], dtype=img.dtype)
            std = np.array([0.229, 0.224, 0.225], dtype=img.dtype)
            img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
            img = np.clip(img, 0.0, 1.0)
            pil = Image.fromarray((img * 255).astype('uint8'))
            return pil
        else:
            # fallback to global min-max normalization (matches pooled-token test_decoder)
            img_min = img.min()
            img_max = img.max()
            img_norm = (img - img_min) / (img_max - img_min + 1e-8)
            img_u8 = (img_norm * 255).astype('uint8')
            pil = Image.fromarray(img_u8)
            return pil


def embedding_to_base64(emb: np.ndarray) -> Tuple[str, Tuple[int, ...], str]:
    """Serialize numpy array to base64 string and return (b64, shape, dtype)"""
    arr = np.asarray(emb)
    # use float16 to reduce size
    arr16 = arr.astype(np.float16)
    b = arr16.tobytes()
    b64 = base64.b64encode(b).decode('ascii')
    return b64, arr16.shape, str(arr16.dtype)


def base64_to_embedding(b64: str, shape: Tuple[int, ...], dtype: str = 'float16') -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
    return arr.astype(np.float32)
