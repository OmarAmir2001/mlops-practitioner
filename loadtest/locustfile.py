import random

from locust import HttpUser, between, task

CONTRACTS = ["month-to-month", "one_year", "two_year"]
INTERNET = ["dsl", "fiber_optic", "no"]
PAYMENTS = [
    "electronic_check",
    "mailed_check",
    "bank_transfer_(automatic)",
    "credit_card_(automatic)",
]
YES_NO = ["yes", "no"]
YES_NO_NOINET = ["yes", "no", "no_internet_service"]


def random_customer() -> dict:
    tenure = random.randint(0, 72)
    monthly = round(random.uniform(18.0, 120.0), 2)
    return {
        "gender": random.choice(["female", "male"]),
        "seniorcitizen": random.choice([0, 1]),
        "partner": random.choice(YES_NO),
        "dependents": random.choice(YES_NO),
        "phoneservice": random.choice(YES_NO),
        "multiplelines": random.choice(["yes", "no", "no_phone_service"]),
        "internetservice": random.choice(INTERNET),
        "onlinesecurity": random.choice(YES_NO_NOINET),
        "onlinebackup": random.choice(YES_NO_NOINET),
        "deviceprotection": random.choice(YES_NO_NOINET),
        "techsupport": random.choice(YES_NO_NOINET),
        "streamingtv": random.choice(YES_NO_NOINET),
        "streamingmovies": random.choice(YES_NO_NOINET),
        "contract": random.choice(CONTRACTS),
        "paperlessbilling": random.choice(YES_NO),
        "paymentmethod": random.choice(PAYMENTS),
        "tenure": tenure,
        "monthlycharges": monthly,
        # correlated, so the customer could actually exist
        "totalcharges": round(monthly * max(tenure, 1), 2),
    }


class ChurnUser(HttpUser):
    wait_time = between(1, 3)

    @task(80)
    def predict_single(self):
        self.client.post("/predict", json=random_customer(), name="/predict")

    @task(15)
    def predict_batch(self):
        batch = [random_customer() for _ in range(random.randint(5, 50))]
        self.client.post("/predict/batch", json=batch, name="/predict/batch")

    @task(5)
    def metadata(self):
        self.client.get("/metadata", name="/metadata")