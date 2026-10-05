from pydantic import BaseModel


class ClassProbability(BaseModel):
    label: str
    name: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_name: str
    confidence: float
    probabilities: list[ClassProbability]
    gradcam_image: str
    model_name: str
    input_size: str
    inference_time_ms: float