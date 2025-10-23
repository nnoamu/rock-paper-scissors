"""
Egységes feature vektor reprezentáció minden feature extractor számára.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum
from .data_object import DataObject


class FeatureType(Enum):
    GEOMETRIC = "geometric"
    DEEP = "deep"
    HYBRID = "hybrid"


@dataclass
class FeatureVector(DataObject):
    feature_type: FeatureType
    extractor_name: str                # -> metadata?
    extraction_time_ms: float          # -> metadata?

    @property
    def features(self) -> np.ndarray:
        return self.data
    
    @property
    def named_features(self) -> Optional[Dict[str, float]]:
        return self.named_data

    @property
    def feature_dimension(self) -> int:
        return self.dimension
    
    @staticmethod
    def description() -> str:
        return "Contains a strictly 1D feature vector."
    
    def __init__(
            self,
            feature_type: FeatureType,
            extractor_name: str,
            extraction_time_ms: float,
            features: np.ndarray,
            feature_dimension: int=1,                                 # bennhagyva kompatibilitás miatt; a DataObject ezt kiszámítja, meg amúgy is 1D vektorról beszéltünk feature vektornak
            named_features: Optional[Dict[str, float]] = None,
            metadata: Optional[Dict[str, Any]] = None
        ):
        if len(features.shape)>1:
            raise TypeError("'features' must be 1 dimensional")

        super().__init__(data=features, is_batch=False, named_data=named_features, metadata=metadata)
        self.feature_type=feature_type
        
        self.extractor_name=extractor_name
        self.extraction_time_ms=extraction_time_ms

    def get_feature_summary(self) -> str:
        return super().get_summary()+(f" | {self.extraction_time_ms:.2f}ms")
    
    def get_summary(self) -> str:
        return self.get_feature_summary()