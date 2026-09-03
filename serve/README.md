# USF-MAE serve

This folder contains a minimal FastAPI wrapper to serve the USF-MAE encoder and decoder using the facebookresearch/mae implementation.

Quick start (CPU):

1. Place the `mae` repo next to this `serve` folder (repo root):

```
git clone https://github.com/facebookresearch/mae mae
```

2. Download the checkpoint `.pt` into a path and set env var `MAE_CHECKPOINT`.

3. Install requirements and run:

```bash
pip install -r requirements.txt
MAE_CHECKPOINT=/path/to/mae_checkpoint.pt python -m uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /encode` — multipart file upload `file`, query param `pooled` (bool). Returns `embedding_b64`, `shape`, `dtype`.
- `POST /decode` — JSON payload `{embedding_b64, shape, dtype}` and returns `image_b64` (PNG base64).

Notes:
- `decode` currently expects per-patch embeddings (not a compact pooled vector). See `serve/inference.py` for details.
- This is a minimal skeleton to test on CPU; further production hardening, batching, and sizing are required before deploying to AWS.
