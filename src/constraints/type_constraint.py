"""
Az adat elemeinek típustát megkötő korlátozás.
"""

import numpy as np
from typing import Final, Type
from core.data_object import DataObject
from core.base_constraint import Constraint

class TypeConstraint(Constraint):

    def __init__(self, type: Type):
        self.__required_type: Final[Type]=type

    def check(self, data: DataObject) -> bool:
        return data.dtype==self.__required_type
    
    def get_required_type(self) -> Type:
        return self.__required_type
    
    @property
    def description(self) -> str:
        return super().description+": The data type must be ["+str(self.__required_type)+"]."