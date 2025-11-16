"""
2 szintű, korlátozott küszöbölést végrehajtó modul.
Bemenet 1. kép: szürkeárnyalatos kép.
Bemenet 2. kép: bináris kép (maszk)
Kimenet:
Az 1. képen végrehajtunk egy flood fill-t:
- a "max intenzitás"*upper_thresh-t meghaladó intenzitásokból indulva
- csak a "max intenzitás"*lower_thresh-nél nagyobb intenzitásokra lépve
- a maszk nem 0 elemeit elkerülve
"""

import cv2
import numpy as np
from core import PreprocessingModule, DataObject
from constraints import TypeConstraint, DimensionConstraint

class ThresholdFillModule(PreprocessingModule):

    def __init__(self, lower_thresh: float=0.2, upper_thresh: float=0.8):
        super().__init__(name="GaussianBlur")
        self.add_constraint(TypeConstraint(np.uint8))
        self.add_constraint(DimensionConstraint(dim=0, length=2))

        self.lower_thresh=lower_thresh
        self.upper_thresh=upper_thresh

    def _process(self, input: DataObject) -> DataObject:
        img=input.data[0]
        barrier=input.data[1]

        img=img*(barrier==0)
        max_intensity=np.max(img)

        current=255*np.array(img>max_intensity*self.upper_thresh, dtype=np.uint8)
        fillable=np.array(img>max_intensity*self.lower_thresh, dtype=np.bool)

        change=True
        structuring_element=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        while change:
            old=current
            current=cv2.dilate(current, structuring_element)
            current[fillable==0]=0
            change=(old!=current).any()
        
        return DataObject(current)
