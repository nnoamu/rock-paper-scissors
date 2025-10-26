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
    def __validate_input(self, data: DataObject) -> Tuple[bool, List[str]]:
        success=True
        msg=[]

        for constraint in self.__constraints:
            if not constraint.check(data):
                success=False
                msg.append(constraint.description)

        return (success, msg)
    
    @abstractmethod
    def _process(self, input: DataObject) -> DataObject | List[DataObject]:
        pass

    @final
    def process(self, input: DataObject | List[DataObject]) -> DataObject | List[DataObject]:
        def process_single(input: DataObject) -> DataObject | List[DataObject]:
            success, msg=self.__validate_input(input)
            if not success:
                raise ConstraintViolationException(self.name, msg)
            return self._process(input)
        
        if isinstance(input, DataObject):
            return process_single(input)
        elif isinstance(input, list):
            all=[]
            for x in input:
                if not isinstance(x, DataObject):
                    raise Exception("Error in "+self.name+": List element is not DataObject")
                x=process_single(x)

                if isinstance(x, DataObject):
                    all.append(x)
                else:   #list[DataObject]
                    all.extend(x)
            return all
        #else:
        raise Exception("Error in"+self.name+": Invalid input type.")
    
    @final
    def add_constraint(self, constraint: Constraint):
        self.__constraints.append(constraint)
    
    @final
    def get_constraints(self) -> List[Constraint]:
        return self.__constraints

    def __str__(self) -> str:
        return self.name