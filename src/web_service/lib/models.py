# Pydantic models for the web service
from pydantic import BaseModel


class InputData(BaseModel):
    Sex: str
    Length: float
    Diameter: float
    Height: float
    Whole_weight: float
    Shucked_weight: float
    Viscera_weight: float
    Shell_weight: float
    Rings: int


class OutputData(BaseModel):
    Age: float
