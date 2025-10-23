"""
Grayscale konverzió modul.
Színes képet szürkeskálássá alakít. 1 csatornás képet ad vissza.
(Base Modulból származtatott modul minta implementáció.)
"""

import cv2
import numpy as np
from core import BaseModule, DataObject
from constraints import NoBatchConstraint, TypeConstraint


class GrayscaleConverterDemo(BaseModule):

    def __init__(self):
        super().__init__(name="Grayscale")
        self.addConstraint(NoBatchConstraint())
        self.addConstraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> DataObject:
        if len(input.shape) == 3 and input.shape[2] == 3:
            return DataObject(cv2.cvtColor(input.data, cv2.COLOR_BGR2GRAY), False)
        return input