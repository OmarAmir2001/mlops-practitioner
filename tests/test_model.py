import numpy as np
import pytest


def test_predict(predictor, sample_customer):
    predictor.model.predict.return_value = np.array([0.73])
    result = predictor.predict(sample_customer)
    assert isinstance(result, dict)
    assert result["churn_probability"] == 0.73
    assert result["churn"] is True
    assert result["threshold"] == 0.5


def test_predict_probability_is_float(predictor, sample_customer):
    predictor.model.predict.return_value = np.array([0.73])
    result = predictor.predict(sample_customer)
    assert isinstance(result["churn_probability"], float)


def test_predict_probability_in_valid_range(predictor, sample_customer):
    predictor.model.predict.return_value = np.array([0.73])
    result = predictor.predict(sample_customer)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_is_deterministic(predictor, sample_customer):
    predictor.model.predict.return_value = np.array([0.73])
    assert predictor.predict(sample_customer) == predictor.predict(sample_customer)


def test_predict_batch_returns_one_result_per_row(predictor, sample_customer):
    predictor.model.predict.return_value = np.array([0.73, 0.21, 0.55])
    results = predictor.predict_batch([sample_customer] * 3)
    assert len(results) == 3
    assert [r["churn"] for r in results] == [True, False, True]


def test_batch_matches_single(predictor, sample_customer):
    """The batch path must agree with the single path."""
    predictor.model.predict.return_value = np.array([0.73])
    single = predictor.predict(sample_customer)
    batch = predictor.predict_batch([sample_customer])
    assert batch[0]["churn_probability"] == single["churn_probability"]


@pytest.mark.parametrize(
    "probability,expected_churn",
    [
        (0.0, False),
        (0.49, False),
        (0.5, True),
        (0.51, True),
        (1.0, True),
    ],
)
def test_churn_threshold_boundary(
    predictor, sample_customer, probability, expected_churn
):
    predictor.model.predict.return_value = np.array([probability])
    result = predictor.predict(sample_customer)
    assert result["churn"] is expected_churn


def test_get_metadata_shape(predictor):
    predictor.model.metadata.run_id = "abc123"
    predictor.model.metadata.utc_time_created = "2026-09-04 00:00:00"
    predictor.model.metadata.model_size_bytes = 3331
    predictor.model.unwrap_python_model.return_value.framework = "logistic_regression"

    meta = predictor.get_metadata()
    assert meta["run_id"] == "abc123"
    assert meta["framework"] == "logistic_regression"
    assert meta["model_name"] == "churn-predictor"
