"""
Bináris képben lévő lyukakat eltüntető modul.
"""

import cv2
import numpy as np
from core import PreprocessingModule, DataObject
from constraints import TypeConstraint

class HoleClosingModule(PreprocessingModule):

    def __init__(self):
        super().__init__(name="HoleClosing")
        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> DataObject:
        img=input.data
        
        ksize=int(np.max(img.shape)/100)
        if ksize%2==0:
            ksize=ksize+1
        
        structuring_element=cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        img=cv2.morphologyEx(img, cv2.MORPH_CLOSE, structuring_element, iterations=2)

        contour, _=cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contour, -1, 255, -1)

        return DataObject(img)
