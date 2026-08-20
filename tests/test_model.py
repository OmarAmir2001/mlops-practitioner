import pytest

def test_predict(predictor):
    predictor.model.predict.return_value = [0.73]
    result =predictor.predict({"gender": "female", "tenure": 1, "monthlycharges": 29.85})
    assert isinstance(result, dict)
    assert "churn_probability" in result
    assert result["churn_probability"] == 0.73
    assert "churn" in result
    assert result["churn"] is True

def test_predict_probability_is_float(predictor):
    predictor.model.predict.return_value = [0.73]
    result = predictor.predict({"gender": "female", "tenure": 1, "monthlycharges": 29.85})
    assert isinstance(result["churn_probability"], float)

def test_predict_probability_in_valid_range(predictor):
    predictor.model.predict.return_value = [0.73]
    result = predictor.predict({"gender": "female", "tenure": 1, "monthlycharges": 29.85})
    assert 0.0 <= result["churn_probability"] <= 1.0

def test_predict_is_deterministic(predictor):
    predictor.model.predict.return_value = [0.73]
    result1 = predictor.predict({"gender": "female", "tenure": 1, "monthlycharges": 29.85})
    result2 = predictor.predict({"gender": "female", "tenure": 1, "monthlycharges": 29.85})
    assert result1 == result2

@pytest.mark.parametrize("probability,expected_churn", [
    (0.0,  False),
    (0.49, False),
    (0.5,  True),   # boundary — inclusive per your >= logic
    (0.51, True),
    (1.0,  True),
])
def test_churn_threshold_boundary(predictor, probability, expected_churn):
    predictor.model.predict.return_value = [probability]
    result = predictor.predict({"gender": "female", "tenure": 1, "monthlycharges": 29.85})
    assert result["churn"] is expected_churn
