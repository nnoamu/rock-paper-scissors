"""
Osztályozó modulok ősosztálya.

Példa használat:
    class MyClassifier(BaseClassifier):
        def classify(self, features: FeatureVector) -> ClassificationResult:
            return ClassificationResult(
                predicted_class=GestureClass.ROCK,
                confidence=0.95,
                class_probabilities={...},
                processing_time_ms=5.2,
                classifier_name=self.name
            )
"""

from abc import ABC, abstractmethod
from .feature_vector import FeatureVector
from .classification_result import ClassificationResult


class BaseClassifier(ABC):

    def __init__(self, name: str):
        self.name = name
        self.is_trained = True

    @abstractmethod
    def classify(self, features: FeatureVector) -> ClassificationResult:
        pass

    def __str__(self) -> str:
        return self.name