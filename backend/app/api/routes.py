from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import ALLOWED_TYPES, MAX_UPLOAD_MB
from app.schemas.prediction import PredictionResponse
from app.services import predictor
from app.services.preprocessing import load_image

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Format non supporté (JPG ou PNG)")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "Image trop volumineuse")

    try:
        img = load_image(data)
    except Exception:
        raise HTTPException(400, "Image illisible")

    return predictor.predict(img)