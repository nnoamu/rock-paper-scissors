"""
Az adat osztályát megkötő korlátozás.
"""

from typing import Final, Type
from core.data_object import DataObject
from core.base_constraint import Constraint

class ClassConstraint(Constraint):

    def __init__(self, classType: Type):
        self.__requiredClass: Final[Type]=classType

    def check(self, data: DataObject) -> bool:
        return isinstance(data, self.__requiredClass)
    
    def getRequiredClass(self) -> Type:
        return self.__requiredClass
    
    @property
    def description(self) -> str:
        return super().description+": Az adat osztálya ["+str(self.__requiredClass)+"], vagy annak leszármazottja kell legyen."