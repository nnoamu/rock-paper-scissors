"""
Feature extraction modulok ősosztálya.

Példa használat:
    class MyExtractor(BaseFeatureExtractor):
        def extract(self, preprocessed_image: np.ndarray) -> FeatureVector:
            features = np.array([1.0, 2.0, 3.0])
            return FeatureVector(
                feature_type=FeatureType.GEOMETRIC,
                extractor_name=self.name,
                extraction_time_ms=10.5,
                features=features,
                feature_dimension=3
            )

        def get_feature_dimension(self) -> int:
            return 3
"""

from abc import ABC, abstractmethod
import numpy as np
from .feature_vector import FeatureVector


class BaseFeatureExtractor(ABC):

    def __init__(self, name: str):
        self.name = name
        self.is_initialized = True

    @abstractmethod
    def extract(self, preprocessed_image: np.ndarray) -> FeatureVector:
        pass

    @abstractmethod
    def get_feature_dimension(self) -> int:
        pass

    def __str__(self) -> str:
        return self.name