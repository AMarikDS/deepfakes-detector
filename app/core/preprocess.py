import numpy as np
from PIL import Image as PILImage

from app.services.logger import logger


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0, posinf=255, neginf=0)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr *= 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        arr = (arr.astype(np.float32) / np.iinfo(arr.dtype).max) * 255
        return np.clip(arr, 0, 255).astype(np.uint8)
    return np.clip(arr.astype(np.float32), 0, 255).astype(np.uint8)


def normalize_image_to_rgb(img_in):
    if isinstance(img_in, PILImage.Image):
        img = img_in
    else:
        img = PILImage.fromarray(_to_uint8(np.asarray(img_in)))

    try:
        import PIL.ImageOps as ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode == "RGBA":
        bg = PILImage.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        img = bg

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img
