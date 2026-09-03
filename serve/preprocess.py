from PIL import Image
from .inference import preprocess_image


def load_image_bytes(b: bytes) -> Image.Image:
    img = Image.open(BytesIO(b))
    return img


def preprocess_pil(img: Image.Image):
    return preprocess_image(img)
