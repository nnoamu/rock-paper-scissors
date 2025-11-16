"""
Gauss simítást végző modul.
Bemenet 2D kép.
Kimenet ugyanolyan alakú simított uint8 kép.
"""

import cv2
import numpy as np
from core import PreprocessingModule, DataObject

class GaussianBlurModule(PreprocessingModule):

    def __init__(self):
        super().__init__(name="GaussianBlur")

    def _process(self, input: DataObject) -> DataObject:
        kernel_size=round(max(input.shape)/20)
        if kernel_size%2==0:
            kernel_size=kernel_size+1
        
        blur=cv2.GaussianBlur(input.data, (kernel_size, kernel_size), 0)
        return DataObject(np.array(blur, dtype=np.uint8))