"""
Egységes feature vektor reprezentáció minden feature extractor számára.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from enum import Enum


class FeatureType(Enum):
    GEOMETRIC = "geometric"
    DEEP = "deep"
    HYBRID = "hybrid"


@dataclass
class FeatureVector:
    feature_type: FeatureType
    extractor_name: str
    extraction_time_ms: float
    features: np.ndarray
    feature_dimension: int
    named_features: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def get_feature_summary(self) -> str:
        return f"{self.feature_type.value} | {self.feature_dimension}D | {self.extraction_time_ms:.2f}ms"