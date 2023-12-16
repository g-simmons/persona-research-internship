from pydantic import BaseModel
from typing import Optional, List
from enum import Enum




class TargetSentimentLevel(Enum):
    EXTREMELY_NEGATIVE = -1
    VERY_NEGATIVE = -0.75
    NEGATIVE = -0.5
    SLIGHTLY_NEGATIVE = -0.25
    NEUTRAL = 0
    SLIGHTLY_POSITIVE = 0.25
    POSITIVE = 0.5
    VERY_POSITIVE = 0.75
    EXTREMELY_POSITIVE = 1


class LLMEditedReview(BaseModel):
    original_acting: str
    original_direction: str
    original_cinematography: str
    edited_component: str
    edited_acting: str
    edited_direction: str
    edited_cinematography: str
    target_sentiment_level: float
    edit_model_name: str