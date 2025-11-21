"""
Bináris képen lévő alakzat Hu-momentumát kinyerő modul.
"""

import time
import math
import cv2
import numpy as np
from typing import List
from core import BaseFeatureExtractor, DataObject, FeatureVector, FeatureType
from constraints import TypeConstraint

class HuMomentsExtractor(BaseFeatureExtractor):

    def __init__(self):
        super().__init__(name="HuMoments")
        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> FeatureVector:
        start=time.perf_counter()

        img=input.data
        features=[-1*math.copysign(1.0, x)*math.log10(abs(x)) for x in cv2.HuMoments(cv2.moments(img)).flatten()]
        result=FeatureVector(FeatureType.GEOMETRIC, self.name, -1, np.array(features))
        
        process_time=(time.perf_counter()-start)*1000
        result.extraction_time_ms=process_time
        return result

    def get_feature_dimension(self) -> int:
        return 1