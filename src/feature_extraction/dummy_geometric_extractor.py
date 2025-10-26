"""
Egyszerű dummy feature extractor teszteléshez.
3 alap jellemzőt számol: átlagos fényerő, szélesség, magasság.
"""

import cv2
import numpy as np
import time

from core.base_feature_extractor import BaseFeatureExtractor
from core.feature_vector import FeatureVector, FeatureType
from core import DataObject


class DummyGeometricExtractor(BaseFeatureExtractor):

    FEATURE_DIMENSION = 3

    def __init__(self):
        super().__init__(name="Dummy_Geometric")

    def _process(self, input: DataObject) -> FeatureVector:
        start_time = time.perf_counter()

        if len(input.shape) == 3:
            gray = cv2.cvtColor(input.data, cv2.COLOR_BGR2GRAY)
        else:
            gray = input.data

        avg_brightness = np.mean(gray) / 255.0
        height, width = gray.shape[:2]
        norm_width = width / 640.0
        norm_height = height / 480.0

        features = np.array([avg_brightness, norm_width, norm_height])

        named_features = {
            'avg_brightness': avg_brightness,
            'norm_width': norm_width,
            'norm_height': norm_height
        }

        end_time = time.perf_counter()
        extraction_time = (end_time - start_time) * 1000

        return FeatureVector(
            feature_type=FeatureType.GEOMETRIC,
            extractor_name=self.name,
            extraction_time_ms=extraction_time,
            features=features,
            feature_dimension=self.FEATURE_DIMENSION,
            named_features=named_features,
            metadata={'image_shape': input.shape}
        )

    def get_feature_dimension(self) -> int:
        return self.FEATURE_DIMENSION