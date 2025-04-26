from pydantic import BaseModel
from typing import Optional

class RaportModel(BaseModel):
    subject: str
    body: str
    metadata: dict
    prediction_model: str
    confidence_model: float
    prediction_rag: str
    confidence_rag: float
    body: Optional[str] = None
