"""
Egységes osztályozási eredmény container minden classifier számára.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class GestureClass(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    predicted_class: GestureClass
    confidence: float
    class_probabilities: Dict[GestureClass, float]
    processing_time_ms: float
    classifier_name: str
    metadata: Optional[Dict] = None

    def get_confidence_percentage(self) -> float:
        return self.confidence * 100