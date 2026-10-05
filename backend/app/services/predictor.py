import time

import numpy as np

from app.core.config import CLASS_NAMES, IMG_SIZE
from app.schemas.prediction import ClassProbability, PredictionResponse
from app.services.model_loader import get_classes, get_model


def predict(img: np.ndarray) -> PredictionResponse:
    model = get_model()
    classes = get_classes()

    start = time.perf_counter()
    probs = model(img[None], training=False).numpy()[0]
    elapsed = (time.perf_counter() - start) * 1000

    ranked = sorted(
        (
            ClassProbability(label=c, name=CLASS_NAMES[c], probability=float(p))
            for c, p in zip(classes, probs)
        ),
        key=lambda x: x.probability,
        reverse=True,
    )
    top = ranked[0]

    return PredictionResponse(
        predicted_class=top.label,
        predicted_name=top.name,
        confidence=top.probability,
        probabilities=ranked,
        model_name="MobileNetV2 (Fine-Tuning)",
        input_size=f"{IMG_SIZE} × {IMG_SIZE} px",
        inference_time_ms=round(elapsed, 1),
    )