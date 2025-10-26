"""
Színcsatorna mentén szétosztó modul.
Színes képet 3 szürkeskálás képpé alakít (r, g, b) sorrendben.
"""

from typing import List
import cv2
import numpy as np
from core.base_processor import PreprocessingModule
from core import DataObject
from constraints import TypeConstraint


class ChannelSplitter(PreprocessingModule):

    def __init__(self):
        super().__init__(name="ChannelSplitter")
        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> List[DataObject]:
        r, g, b=cv2.split(input.data)
        return [DataObject(r), DataObject(g), DataObject(b)]