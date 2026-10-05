import base64
import io
from functools import lru_cache

import numpy as np
import tensorflow as tf
from PIL import Image

from app.services.model_loader import get_model

OUT_SIZE = 256
ALPHA = 0.4


@lru_cache
def _parts():
    model = get_model()
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    conv_model = tf.keras.Model(base.input, base.get_layer("out_relu").output)
    head = model.layers[model.layers.index(base) + 1:]
    return conv_model, head


def _jet(x: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam_overlay(img: np.ndarray, class_idx: int) -> str:
    conv_model, head = _parts()
    x = tf.convert_to_tensor(img[None], tf.float32) / 127.5 - 1.0

    with tf.GradientTape() as tape:
        conv = conv_model(x)
        tape.watch(conv)
        h = conv
        for layer in head:
            h = layer(h)
        score = h[:, class_idx]

    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC)
    heat = _jet(np.asarray(cam_img, dtype=np.float32) / 255) * 255

    base_img = Image.fromarray(img.astype("uint8")).resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC)
    out = np.uint8((1 - ALPHA) * np.asarray(base_img, dtype=np.float32) + ALPHA * heat)

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()