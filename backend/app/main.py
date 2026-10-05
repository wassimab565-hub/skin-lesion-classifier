from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import CORS_ORIGINS
from app.services.model_loader import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="Skin Lesion Classifier API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)