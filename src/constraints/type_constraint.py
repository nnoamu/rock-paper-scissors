"""
Az adat elemeinek típustát megkötő korlátozás.
"""

import numpy as np
from typing import Final, Type
from core.data_object import DataObject
from core.base_constraint import Constraint

class TypeConstraint(Constraint):

    def __init__(self, type: Type):
        self.__requiredType: Final[Type]=type

    def check(self, data: DataObject) -> bool:
        return data.dtype==self.__requiredType
    
    def getRequiredType(self) -> Type:
        return self.__requiredType
    
    @property
    def description(self) -> str:
        return super().description+": Az adat elemeinek típusa ["+str(self.__requiredType)+"] kell legyen."