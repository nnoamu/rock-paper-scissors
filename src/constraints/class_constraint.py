"""
Az adat osztályát megkötő korlátozás.
"""

from typing import Final, Type
from core.data_object import DataObject
from core.base_constraint import Constraint

class ClassConstraint(Constraint):

    def __init__(self, classType: Type):
        self.__required_class: Final[Type]=classType

    def check(self, data: DataObject) -> bool:
        return isinstance(data, self.__required_class)
    
    def get_required_class(self) -> Type:
        return self.__required_class
    
    @property
    def description(self) -> str:
        return super().description+": The data class must be or be derived from ["+str(self.__required_class)+"]."