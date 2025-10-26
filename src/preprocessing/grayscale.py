"""
Grayscale konverzió modul.
Színes képet szürkeskálássá alakít, majd visszakonvertálja BGR-re a konzisztencia miatt.
"""

from typing import List
import cv2
import numpy as np
from core.base_processor import PreprocessingModule
from core import DataObject
from constraints import TypeConstraint


class GrayscaleConverter(PreprocessingModule):

    def __init__(self):
        super().__init__(name="Grayscale")
        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> DataObject:
        gray=cv2.cvtColor(input.data, cv2.COLOR_BGR2GRAY) if len(input.shape)==3 and input.shape[2]==3 else input.data
        result=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape)==2 else gray
        return DataObject(result)