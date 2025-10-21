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
    #extractor_name: str                # -> metadata
    #extraction_time_ms: float          # -> metadata
    #features: np.ndarray
    #feature_dimension: int
    #named_features: Optional[Dict[str, float]] = None
    #metadata: Dict[str, Any] = {}

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
            features: np.ndarray[Tuple[int], np.dtype],
            feature_dimension: int,                                 # bennhagyva kompatibilitás miatt; a DataObject ezt kiszámítja
            named_features: Optional[Dict[str, float]] = None,
            metadata: Optional[Dict[str, Any]] = {}
        ):
        super().__init__(data=features, is_batch=False, named_data=named_features, metadata=metadata)
        self.feature_type=feature_type

        if self.metadata is None:
            self.metadata={}
        self.metadata["extractor_name"]=extractor_name
        self.metadata["extraction_time_ms"]=extraction_time_ms

    def get_feature_summary(self) -> str:
        return super().get_summary()+(f" | {self.metadata['extraction_time_ms']:.2f}ms" if self.metadata is not None else "")
    
    def get_summary(self) -> str:
        return self.get_feature_summary()