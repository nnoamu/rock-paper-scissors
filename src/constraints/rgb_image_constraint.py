"""
RGB kép formájú adatot megkövetelő korlátozás. 3 csatornás képeket követel meg.
"""

from core.data_object import DataObject
from core.base_constraint import Constraint

class RGBImageConstraint(Constraint):

    def check(self, data: DataObject) -> bool:
        return len(data.shape)==3 and data.shape[2]==3
    
    @property
    def description(self) -> str:
        return super().description+": Only RGB image-shaped data (height, width, 3) is allowed. The data dimension must be 3 and the data must consist of 3 channels (final dimension)."