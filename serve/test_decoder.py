"""Quick test: decode a pooled embedding back to an image and save it.

This script:
- loads the MAE model from checkpoint
- encodes a sample image to obtain the pooled embedding
- constructs a token sequence from that pooled vector
- runs the model decoder and reconstructs an image
- saves the output as `decoded_from_case001.png` in the repo root
"""
import os
from PIL import Image
import numpy as np
import torch

from serve.inference import load_model, encode_image


def main():
    ckpt = os.environ.get('MAE_CHECKPOINT', './USF-MAE_full_pretrain_43dataset_100epochs.pt')
    model = load_model(ckpt, device='cpu')
    model.eval()

    img_path = 'dataset_breast/BrEaST-Lesions_USG-images_and_masks/case001.png'
    img = Image.open(img_path)

    # get pooled embedding (1, D)
    emb = encode_image(img, model, device='cpu', pooled=True)
    emb_t = torch.from_numpy(emb).float()  # (1, D)

    # construct token sequence: cls + all patches (repeat pooled vector)
    num_patches = model.patch_embed.num_patches
    embed_dim = model.cls_token.shape[-1]

    # create sequence of length num_patches+1
    x_tokens = emb_t.unsqueeze(1).repeat(1, num_patches + 1, 1)  # (1, num_patches+1, D)

    # ids_restore identity (no shuffling)
    ids_restore = torch.arange(num_patches).unsqueeze(0)

    # run decoder
    with torch.no_grad():
        try:
            recon_patches = model.forward_decoder(x_tokens, ids_restore)  # (B, N, p*p*3)
        except TypeError:
            # fallback: some implementations use different decoder name
            recon_patches = model.decoder(x_tokens)

    # reconstruct image
    try:
        imgs = model.unpatchify(recon_patches)
    except Exception:
        # If decoder returned already-unpatched image-like tensor
        imgs = recon_patches

    img_tensor = imgs[0].cpu().numpy()  # (3, H, W)
    # normalize to 0-255 for saving
    img_min = img_tensor.min()
    img_max = img_tensor.max()
    img_norm = (img_tensor - img_min) / (img_max - img_min + 1e-8)
    img_u8 = (img_norm * 255).astype('uint8')
    img_u8 = np.transpose(img_u8, (1, 2, 0))
    pil = Image.fromarray(img_u8)

    out_path = os.path.join(os.getcwd(), 'decoded_from_case001.png')
    pil.save(out_path)
    print('Saved decoded image to', out_path)


if __name__ == '__main__':
    main()
