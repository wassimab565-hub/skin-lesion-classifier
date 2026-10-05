import json
from functools import lru_cache

from tensorflow import keras

from app.core.config import CLASSES_PATH, MODEL_PATH


@lru_cache
def get_model():
    return keras.models.load_model(MODEL_PATH)


@lru_cache
def get_classes() -> list[str]:
    return json.loads(CLASSES_PATH.read_text())