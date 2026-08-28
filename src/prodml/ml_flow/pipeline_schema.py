import yaml
from pathlib import Path
from typing import Dict, Any, Literal
from pydantic import BaseModel
from prodml.config import get_settings
import structlog
log = structlog.get_logger(__name__)

# 1. load settings
settings = get_settings()

# 2. Define sub-schemas for ML
class CVSettings(BaseModel):
    cv_folds: int
    scoring_metric: str

# 3. The Combined Configuration Schema (Env + YAML)
class CompleteMLConfig(BaseModel):
    # Variables coming from the .env file
    tracking_uri: str = settings.MLFLOW_TRACKING_URI
    
    # Variables coming from the YAML file
    active_model: Literal["logistic", "xgboost", "random_forest", "mlp"]
    cross_validation: CVSettings
    models: Dict[str, Dict[str, Any]]

# 4. Loader function that merges them
def load_ml_pipeline() -> CompleteMLConfig:
    # Read the YAML file path out of the environment variables
    yaml_path_str = settings.ML_CONFIG_PATH
    yaml_path = Path(yaml_path_str)
    
    if not yaml_path.exists():
        raise log.error("yaml_path_not_found", path=yaml_path_str)
        
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
        log.info("yaml_loaded", path=yaml_path_str)
        
    # Unpack the YAML dictionary into Pydantic to combine with the Env fields
    return CompleteMLConfig(**yaml_data)
