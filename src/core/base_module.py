"""
Az összes modul közös ősosztálya.
Támogatja a batch inference optimalizációt: ha a modul implementálja a _process_batch() metódust,
akkor azt használja batch esetén.
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

    def _process_batch(self, inputs: List[DataObject]) -> List[DataObject | List[DataObject]]:
        raise NotImplementedError(f"{self.name} does not implement batch processing")

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
            for x in input:
                if not isinstance(x, DataObject):
                    raise Exception("Error in "+self.name+": List element is not DataObject")
                success, msg = self.__validate_input(x)
                if not success:
                    raise ConstraintViolationException(self.name, msg)

            try:
                results = self._process_batch(input)
                all = []
                for result in results:
                    if isinstance(result, DataObject):
                        all.append(result)
                    elif isinstance(result, list):
                        all.extend(result)
                    else:
                        raise Exception(f"Error in {self.name}: Invalid batch result type")
                return all
            except NotImplementedError:
                all=[]
                for x in input:
                    x=process_single(x)

                    if isinstance(x, DataObject):
                        all.append(x)
                    else:
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