"""
Éldetektálást végző modul.
Bemenet: Kép.
Kimenet: Bináris élkép. 
"""

import cv2
import numpy as np
from core import PreprocessingModule, DataObject
from constraints import TypeConstraint

MAX_IMAGE_SIZE=300

class EdgeDetectorModule(PreprocessingModule):

    def __init__(self, lower_thresh: int, upper_thresh: int):
        super().__init__(name="EdgeDetector")
        self.upper_thresh=upper_thresh
        self.lower_thresh=lower_thresh

        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> DataObject:
        data=input.data
        shape=input.shape[:2]

        mx=max(shape)

        is_shrunk=False
        if mx>MAX_IMAGE_SIZE:
            is_shrunk=True
            tg=MAX_IMAGE_SIZE
            r=tg/mx
            new_w=round(data.shape[0]*r)
            new_h=round(data.shape[1]*r)
            data=cv2.resize(data, (new_w, new_h))
        
        mx=max(data.shape)
        kernel_size=round(mx/20)
        if kernel_size%2==0:
            kernel_size=kernel_size+1

        data=cv2.GaussianBlur(data, (kernel_size, kernel_size), 0)
        data=cv2.GaussianBlur(data, (kernel_size, kernel_size), 0)
        
        data=cv2.Canny(data, self.lower_thresh, self.upper_thresh)
        data=cv2.dilate(data, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        if is_shrunk:
            data=cv2.resize(data, (shape[1], shape[0]))

        return DataObject(data)