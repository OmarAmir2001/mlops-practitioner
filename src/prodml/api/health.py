import structlog
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from ..config import Settings, get_settings

log = structlog.get_logger(__name__)

base_router = APIRouter(prefix="", tags=["Base Routes"])


@base_router.get("/health")
def health_check(request: Request, app_settings: Settings = Depends(get_settings)):
    """Health check endpoint."""
    model = request.app.state.model

    if model is None:
        log.error("health_check_failed", reason="model_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Model not loaded"},
        )

    log.info("health_check_ok")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "app_name": app_settings.APP_NAME,
            "app_version": app_settings.APP_VERSION,
        },
    )


@base_router.get("/metadata")
def metadata_check(request: Request, app_settings: Settings = Depends(get_settings)):
    """Metadata check endpoint."""
    model = request.app.state.model

    if model is None:
        log.error("metadata_check_failed", reason="model_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Model not loaded"},
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content=model.get_metadata())
