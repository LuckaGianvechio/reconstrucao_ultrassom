from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os
import base64
from io import BytesIO
from PIL import Image

from .inference import load_model, encode_image, decode_embedding, embedding_to_base64, base64_to_embedding


app = FastAPI(title='USF-MAE Serve (encoder/decoder)')

# Load model on startup if env var provided
MODEL = None
MODEL_DEVICE = os.getenv('MODEL_DEVICE', 'cpu')
CHECKPOINT = os.getenv('MAE_CHECKPOINT', '')
MAE_PATH = os.getenv('MAE_REPO_PATH', os.path.join(os.path.dirname(__file__), '..', 'mae'))


@app.on_event('startup')
def startup_event():
    global MODEL
    if CHECKPOINT:
        MODEL = load_model(CHECKPOINT, device=MODEL_DEVICE, mae_repo_path=MAE_PATH)


class DecodePayload(BaseModel):
    embedding_b64: str
    shape: tuple
    dtype: str = 'float16'
    ids_restore: list = None


@app.post('/encode')
async def encode(file: UploadFile = File(...), pooled: bool = False):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image: {e}')

    if MODEL is None:
        raise HTTPException(status_code=500, detail='Model not loaded. Set MAE_CHECKPOINT env var.')

    # default: return per-patch embeddings (pooled=False) for decoder compatibility
    res = encode_image(img, MODEL, device=MODEL_DEVICE, pooled=pooled)
    # encode_image may return (emb, ids_restore) for per-patch outputs
    if isinstance(res, tuple):
        emb, ids = res
    else:
        emb = res
        ids = None
    b64, shape, dtype = embedding_to_base64(emb)
    resp = {'embedding_b64': b64, 'shape': shape, 'dtype': dtype}
    if ids is not None:
        resp['ids_restore'] = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    return JSONResponse(resp)


@app.post('/decode')
async def decode(payload: DecodePayload):
    if MODEL is None:
        raise HTTPException(status_code=500, detail='Model not loaded. Set MAE_CHECKPOINT env var.')

    emb = base64_to_embedding(payload.embedding_b64, tuple(payload.shape), payload.dtype)
    try:
        ids = payload.ids_restore
        img = decode_embedding(emb, MODEL, device=MODEL_DEVICE, ids_restore=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Decode failed: {e}')

    buf = BytesIO()
    img.save(buf, format='PNG')
    b64img = base64.b64encode(buf.getvalue()).decode('ascii')
    return JSONResponse({'image_b64': b64img})
