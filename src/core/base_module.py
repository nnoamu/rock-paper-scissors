"""
Az összes modul közös ősosztálya.
"""

from abc import ABC, abstractmethod
from typing import List, Final, final, Tuple
from .data_object import DataObject
from .base_constraint import Constraint
from .constraint_violation_exception import ConstraintViolationException


class BaseModule(ABC):

    def __init__(self, name: str):
        self.name: Final[str] = name
        self.__constraints: List[Constraint]=[]
    
    @final
    def __validateInput(self, data: DataObject) -> Tuple[bool, List[str]]:
        success=True
        msg=[]

        for constraint in self.__constraints:
            if not constraint.check(data):
                success=False
                msg.append(constraint.description)

        return (success, msg)
    
    @abstractmethod
    def _process(self, input: DataObject) -> DataObject:
        pass

    @final
    def process(self, input: DataObject) -> DataObject:
        success, msg=self.__validateInput(input)
        if not success:
            raise ConstraintViolationException(self.name, msg)
        return self._process(input)
    
    @final
    def addConstraint(self, constraint: Constraint):
        self.__constraints.append(constraint)
    
    @final
    def getConstraints(self) -> List[Constraint]:
        return self.__constraints

    def __str__(self) -> str:
        return self.name