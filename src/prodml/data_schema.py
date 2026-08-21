# api/schema.py
from typing import Literal
from pydantic import BaseModel, Field
"""Pydantic schema for customer data."""

class Customer(BaseModel):
    gender: Literal['female', 'male']
    seniorcitizen: int = Field(ge=0, le=1)
    partner: Literal['yes', 'no']
    dependents: Literal['yes', 'no']
    phoneservice: Literal['yes', 'no']
    multiplelines: Literal['yes', 'no', 'no_phone_service']
    internetservice: Literal['dsl', 'fiber_optic', 'no']
    onlinesecurity: Literal['yes', 'no', 'no_internet_service']
    onlinebackup: Literal['yes', 'no', 'no_internet_service']
    deviceprotection: Literal['yes', 'no', 'no_internet_service']
    techsupport: Literal['yes', 'no', 'no_internet_service']
    streamingtv: Literal['yes', 'no', 'no_internet_service']
    streamingmovies: Literal['yes', 'no', 'no_internet_service']
    contract: Literal['month-to-month', 'one_year', 'two_year']
    paperlessbilling: Literal['yes', 'no']
    paymentmethod: Literal[
        'electronic_check',
        'mailed_check',
        'bank_transfer_(automatic)',
        'credit_card_(automatic)',
    ]
    tenure: int = Field(ge=0, le=100)
    monthlycharges: float = Field(gt=0)
    totalcharges: float = Field(ge=0)

    model_config = {
        'json_schema_extra': {
            'example': {
                'gender': 'female',
                'seniorcitizen': 0,
                'partner': 'yes',
                'dependents': 'no',
                'phoneservice': 'no',
                'multiplelines': 'no_phone_service',
                'internetservice': 'dsl',
                'onlinesecurity': 'no',
                'onlinebackup': 'yes',
                'deviceprotection': 'no',
                'techsupport': 'no',
                'streamingtv': 'no',
                'streamingmovies': 'no',
                'contract': 'month-to-month',
                'paperlessbilling': 'yes',
                'paymentmethod': 'electronic_check',
                'tenure': 1,
                'monthlycharges': 29.85,
                'totalcharges': 29.85,
            }
        }
    }