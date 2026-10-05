from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import CORS_ORIGINS, IMG_SIZE
from app.services.model_loader import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = get_model()
    model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"), training=False)
    yield


app = FastAPI(title="Skin Lesion Classifier API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)