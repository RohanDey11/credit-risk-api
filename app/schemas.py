from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"

class LoanIntent(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"

class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

class DefaultOnFile(str, Enum):
    Y = "Y"
    N = "N"

class CreditRiskInput(BaseModel):
    person_age: int = Field(ge=18, le=100, description="Age 18-100, dropped >100 as error")
    person_income: int = Field(gt=0, le=10000000, description="Annual income >0")
    person_home_ownership: HomeOwnership
    person_emp_length: float = Field(ge=0, le=100)
    loan_intent: LoanIntent
    loan_grade: LoanGrade
    loan_amnt: int = Field(gt=0, le=50000)
    loan_int_rate: float = Field(ge=5, le=30)
    loan_percent_income: float = Field(ge=0, le=1, description="Your #1 feature 0.26 importance")
    cb_person_default_on_file: DefaultOnFile
    cb_person_cred_hist_length: int = Field(ge=0, le=30)

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 32,
                "person_income": 59000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "MEDICAL",
                "loan_grade": "C",
                "loan_amnt": 5500,
                "loan_int_rate": 12.87,
                "loan_percent_income": 0.09,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 3
            }
        }
    }
