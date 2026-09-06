import pytest
from pydantic import ValidationError

from prodml.ml_flow.pipeline_schema import (
    CompleteMLConfig,
    CVSettings,
    SweepRanges,
    load_ml_pipeline,
)

VALID = {
    "active_model": "xgboost",
    "cross_validation": {"cv_folds": 5, "scoring_metric": "f1"},
    "models": {
        "logistic": {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"},
        "xgboost": {"n_estimators": 200, "learning_rate": 0.05},
        "mlp": {"n_layers": 2, "hidden_size": 128},
    },
    "sweep": {
        "n_trials": 10,
        "xgboost": {
            "max_depth": [3, 10],
            "learning_rate": [0.01, 0.3],
            "n_estimators": [50, 300],
            "subsample": [0.6, 1.0],
        },
    },
}


def test_valid_config_parses():
    cfg = CompleteMLConfig(**VALID)
    assert cfg.active_model == "xgboost"
    assert cfg.sweep.n_trials == 10
    assert cfg.sweep.xgboost.max_depth == (3, 10)


def test_unknown_active_model_rejected():
    bad = {**VALID, "active_model": "transformer"}
    with pytest.raises(ValidationError):
        CompleteMLConfig(**bad)


def test_missing_sweep_rejected():
    bad = {k: v for k, v in VALID.items() if k != "sweep"}
    with pytest.raises(ValidationError):
        CompleteMLConfig(**bad)


def test_scalar_range_rejected():
    """max_depth must be a pair, not a single value."""
    bad = {
        **VALID,
        "sweep": {
            **VALID["sweep"],
            "xgboost": {**VALID["sweep"]["xgboost"], "max_depth": 10},
        },
    }
    with pytest.raises(ValidationError):
        CompleteMLConfig(**bad)


def test_cv_settings_types():
    cv = CVSettings(cv_folds=5, scoring_metric="roc_auc")
    assert cv.cv_folds == 5


def test_sweep_ranges_coerce_to_tuples():
    r = SweepRanges(
        max_depth=[3, 10],
        learning_rate=[0.01, 0.3],
        n_estimators=[50, 300],
        subsample=[0.6, 1.0],
    )
    assert isinstance(r.max_depth, tuple)


def test_loader_raises_on_missing_file(monkeypatch, tmp_path):
    from prodml.ml_flow import pipeline_schema

    monkeypatch.setattr(
        pipeline_schema.settings, "ML_CONFIG_PATH", str(tmp_path / "nope.yaml")
    )
    with pytest.raises(FileNotFoundError):
        load_ml_pipeline()


def test_loader_reads_real_config():
    """The committed pipeline_config.yaml must be valid."""
    cfg = load_ml_pipeline()
    assert cfg.tracking_uri.startswith("http")
    assert "xgboost" in cfg.models
