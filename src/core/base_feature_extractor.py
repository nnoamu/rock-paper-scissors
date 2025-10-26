"""
Feature extraction modulok ősosztálya.
Tartalmazott korlátozások:
- ImageLikeConstraint

Példa használat:
    class MyExtractor(BaseFeatureExtractor):
        def _process(self, input: DataObject) -> FeatureVector:
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
from .base_module import BaseModule
from constraints import ImageLikeConstraint
from .data_object import DataObject
from typing import List, cast, final


class BaseFeatureExtractor(BaseModule):

    def __init__(self, name: str):
        super().__init__(name)
        self.add_constraint(ImageLikeConstraint())

        self.is_initialized = True

    @final
    def extract(self, preprocessed_image: DataObject | List[DataObject]) -> FeatureVector | List[FeatureVector]:
        result=self.process(preprocessed_image)
        if (not isinstance(result, FeatureVector)) and ((not isinstance(result, list)) or not isinstance(result[0], FeatureVector)):
            raise Exception("Error in "+self.name+": Output type is not FeatureVector or List[FeatureVector]")
        return cast(FeatureVector | List[FeatureVector], result)

    @abstractmethod
    def get_feature_dimension(self) -> int:
        pass

    def visualize(self, image: np.ndarray, features: FeatureVector) -> np.ndarray:
        return image.copy()

    def __str__(self) -> str:
        return self.name