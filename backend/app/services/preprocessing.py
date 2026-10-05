import io

import numpy as np
import tensorflow as tf
from PIL import Image

from app.core.config import IMG_SIZE


def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return tf.image.resize(np.asarray(img), (IMG_SIZE, IMG_SIZE)).numpy()