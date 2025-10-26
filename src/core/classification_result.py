"""
Egységes osztályozási eredmény container minden classifier számára.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
from .data_object import DataObject
import numpy as np


class GestureClass(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult(DataObject):
    predicted_class: GestureClass
    confidence: float
    class_probabilities: Dict[GestureClass, float]
    processing_time_ms: float
    classifier_name: str

    def __init__(self,
                predicted_class: GestureClass,
                confidence: float,
                class_probabilities: Dict[GestureClass, float],
                processing_time_ms: float,
                classifier_name: str,
                metadata: Optional[Dict] = None
        ):
        super().__init__(np.array([0]), None, metadata)
        self.predicted_class=predicted_class
        self.confidence=confidence
        self.class_probabilities=class_probabilities
        self.processing_time_ms=processing_time_ms
        self.classifier_name=classifier_name

    def get_confidence_percentage(self) -> float:
        return self.confidence * 100