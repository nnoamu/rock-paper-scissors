"""
Képfelosztó modul két játékos módhoz.
1 képet 2 részre oszt (bal és jobb fél), középen vágva.
TODO: ez lehetne ettől intelligensebb, elkerülve a vakon vágást
"""

from typing import List
import numpy as np
from core.base_processor import PreprocessingModule
from core import DataObject
from constraints import TypeConstraint


class ImageSplitterModule(PreprocessingModule):

    def __init__(self):
        super().__init__(name="ImageSplitter")
        self.add_constraint(TypeConstraint(np.uint8))

    def _process(self, input: DataObject) -> List[DataObject]:
        image = input.data
        width = image.shape[1]
        mid_point = width // 2

        left_half = image[:, :mid_point]
        right_half = image[:, mid_point:]

        return [DataObject(left_half), DataObject(right_half)]
