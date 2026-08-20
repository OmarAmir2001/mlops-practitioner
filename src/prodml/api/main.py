from fastapi import FastAPI
from .health import base_router
from .predict import predict_router
from ..config import get_settings
from ..model import ChurnPredictor
from contextlib import asynccontextmanager
from ..logging_conf import configure_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.model = ChurnPredictor()
    yield


app =FastAPI(lifespan=lifespan)
app.include_router(base_router)
app.include_router(predict_router)
