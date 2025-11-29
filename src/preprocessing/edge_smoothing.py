"""
Bináris képen élsimítást végző modul.
"""

import cv2
import numpy as np
from core import PreprocessingModule, DataObject
from constraints import TypeConstraint
from scipy.interpolate import splprep, splev

class EdgeSmoothingModule(PreprocessingModule):

    def __init__(self):
        super().__init__(name="EdgeSmoothing")
        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> DataObject:
        img=input.data
        
        ksize=int(np.max(img.shape)/20)
        if ksize%2==0:
            ksize=ksize+1

        for _ in range(20):
            img=cv2.GaussianBlur(img, (ksize, ksize), 0)
            img=np.array((img>127.5)*255, dtype=np.uint8)

        return DataObject(img)
