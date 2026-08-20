from ..config import get_settings, Settings
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
import structlog
from ..data_schema import Customer

log = structlog.get_logger(__name__)

predict_router = APIRouter(prefix="", tags=["Predictions"])

@predict_router.post("/predict")
def predict_churn(customer: Customer, request: Request):
    model = request.app.state.model
    if model is None:
        log.error("predict_churn_failed", reason="model_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Model not loaded"},
        )
    

    customer_dict = customer.model_dump()
    prediction = model.predict(customer_dict)
    log.info("predict_churn_ok", prediction=prediction)
    return JSONResponse(status_code=status.HTTP_200_OK, content=prediction)


@predict_router.post("/predict/batch")
def predict_churn_batch(customers: list[Customer], request: Request):
    model = request.app.state.model
    if model is None:
        log.error("predict_churn_failed", reason="model_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "message": "Model not loaded"},
        )
    
    customer_dicts = [customer.model_dump() for customer in customers]
    predictions = model.predict_batch(customer_dicts)
    log.info("predict_churn_batch_ok", predictions=predictions)
    return JSONResponse(status_code=status.HTTP_200_OK, content=predictions)