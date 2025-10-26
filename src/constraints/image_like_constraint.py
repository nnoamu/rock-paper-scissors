"""
Kép formájú adatot megkövetelő korlátozás. 1 és többcsatornás képeket is megenged (2-3 dimenziós ndarray-eket)
"""

from core.data_object import DataObject
from core.base_constraint import Constraint

class ImageLikeConstraint(Constraint):

    def check(self, data: DataObject) -> bool:
        return len(data.shape)>1 and len(data.shape)<4
    
    @property
    def description(self) -> str:
        return super().description+": Only image-shaped data is allowed. The data dimension must be 2 (e.g. gray/binary) or 3 (e.g. RGB)."