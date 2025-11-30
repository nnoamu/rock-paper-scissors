"""
Bináris képek 1 komponenseit kivágó modul. A kimenete több kép is lehet.
"""

import cv2
from typing import List
import numpy as np
from core.base_processor import PreprocessingModule
from core import DataObject
from constraints import TypeConstraint


class ObjectSeparatorModule(PreprocessingModule):

    def __init__(self):
        super().__init__(name="ObjectSeparator")
        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> DataObject | List[DataObject]:
        img=input.data
        contours, _=cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result=[]

        for cnt in contours:
            obj=np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(obj, [cnt], 0, 255, -1)
            x, y, w, h=cv2.boundingRect(cnt)
            obj=obj[y:y+h, x:x+w]
            result.append(DataObject(obj))
        
        return result[0] if len(result)==1 else result

        