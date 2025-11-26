"""
Az adat egy dimenzió menti hosszát megkötő korlátozás.
"""

import numpy as np
from typing import Final
from core.data_object import DataObject
from core.base_constraint import Constraint

class DimensionConstraint(Constraint):

    def __init__(self, dim: int, length: int):
        self.__restricted_dim: Final[int]=dim
        self.__required_length: Final[int]=length

    def check(self, data: DataObject) -> bool:
        return data.shape[self.__restricted_dim]==self.__required_length
    
    @property
    def description(self) -> str:
        return super().description+": The data must be "+str(self.__required_length)+" along dimension "+str(self.__restricted_dim)+"."