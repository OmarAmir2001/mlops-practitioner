from ..config import get_settings, Settings
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
import structlog

log = structlog.get_logger(__name__)

base_router = APIRouter(prefix="", tags=["Base Routes"])


@base_router.get("/health")
def health_check(request: Request, app_settings: Settings = Depends(get_settings)):
    model = request.app.state.model

    if model is None:
        log.error("health_check_failed", reason="model_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Model not loaded"},
        )

    log.info("health_check_ok")
    return JSONResponse( status_code= status.HTTP_200_OK ,
                        content={"status": "healthy", "app_name": app_settings.APP_NAME,
                                                    "app_version": app_settings.APP_VERSION})

@base_router.get("/metadata")
def metadata_check(request: Request, app_settings: Settings = Depends(get_settings)):
    model = request.app.state.model
    metadata = request.app.state.metadata

    if model is None:
        log.error("health_check_failed", reason="model_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Model not loaded"},
        )
    if metadata is None:
        log.error("metadata_check_failed", reason="metadata_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Metadata not loaded"},
        )

    
    feature_names = model.dv.get_feature_names_out().tolist()
    model_version = metadata.get_model_version()
    trained_at = metadata.get_trained_at()
    framework = metadata.get_framework()
    artifact_hash = metadata.artifact_hash

    return JSONResponse(status_code=status.HTTP_200_OK, content={"feature_names": feature_names,
                                                                "model_version": model_version,
                                                                "trained_at": trained_at,
                                                                "framework": framework, 
                                                                "artifact_hash": artifact_hash})

    
    
