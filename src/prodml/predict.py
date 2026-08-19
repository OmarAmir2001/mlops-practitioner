import pickle
from config import get_settings

settings = get_settings()

with open(settings.MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print(f'Model loaded from {settings.MODEL_FILE}')


def predict_customer(customer: dict) -> dict:
    """Take one customer dictionary, return the churn probability and decision."""
    X = dv.transform([customer])
    probability = float(model.predict_proba(X)[0, 1])

    return {
        'churn_probability': round(probability, 4),
        'churn': bool(probability >= settings.CHURN_THRESHOLD),
        'threshold': settings.CHURN_THRESHOLD,
    }