"""
schemas.py — Pydantic request/response models for the API.

Field names and types mirror the raw training CSV columns exactly (minus
PREMIUM, minus OBJECT_ID which was only used for grouping during training,
not as a model feature) so a caller's request maps directly onto what
src.predict.clean_and_engineer() expects — no translation layer to keep
in sync.
"""
from typing import Optional

from pydantic import BaseModel, Field


class PolicyInput(BaseModel):
    """One vehicle insurance policy record to price.

    Date fields must match the training data's raw format (DD-MON-YY,
    e.g. "08-AUG-13") — this is intentional: it's the same format
    clean_and_engineer() parses for training data, so there is exactly
    one date-parsing implementation in the whole system.
    """
    SEX: int = Field(..., description="Policyholder sex code as used in training data (0, 1, or 2)")
    INSR_BEGIN: str = Field(..., description="Policy start date, format DD-MON-YY, e.g. '08-AUG-13'")
    INSR_END: str = Field(..., description="Policy end date, format DD-MON-YY, e.g. '07-AUG-14'")
    INSR_TYPE: int = Field(..., description="Insurance type code, e.g. 1201, 1202, 1204")
    INSURED_VALUE: float = Field(..., ge=0, description="Insured value of the vehicle")
    PROD_YEAR: Optional[int] = Field(None, description="Vehicle production year")
    SEATS_NUM: Optional[int] = Field(None, description="Number of seats")
    CARRYING_CAPACITY: Optional[float] = Field(None, description="Carrying capacity, if applicable")
    TYPE_VEHICLE: str = Field(..., description="Vehicle type, e.g. 'Truck', 'Automobile', 'Pick-up'")
    CCM_TON: Optional[float] = Field(None, description="Engine CC or tonnage rating")
    MAKE: str = Field(..., description="Vehicle manufacturer, e.g. 'TOYOTA'")
    USAGE: str = Field(..., description="Vehicle usage category, e.g. 'Private', 'Taxi', 'Own Goods'")

    class Config:
        json_schema_extra = {
            "example": {
                "SEX": 0,
                "INSR_BEGIN": "08-AUG-13",
                "INSR_END": "07-AUG-14",
                "INSR_TYPE": 1202,
                "INSURED_VALUE": 519755.22,
                "PROD_YEAR": 2007,
                "SEATS_NUM": 4,
                "CARRYING_CAPACITY": 6,
                "TYPE_VEHICLE": "Pick-up",
                "CCM_TON": 3153,
                "MAKE": "NISSAN",
                "USAGE": "Own Goods",
            }
        }


class PredictionResponse(BaseModel):
    predicted_premium: float = Field(..., description="Predicted premium in the training data's currency units")
    model_version: str = Field(..., description="Winning model identifier from artifacts/metrics.json")
    flagged: bool = Field(..., description="True if the prediction falls outside the configured sanity bounds")


class BatchPolicyInput(BaseModel):
    records: list[PolicyInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    environment: str
    model_version: str
    artifacts: dict