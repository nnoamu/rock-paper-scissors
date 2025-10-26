"""
Általános keret a modulok megengedett bemenetére vonatkozó korlátozások definiálásához.
"""
from abc import ABC, abstractmethod
from .data_object import DataObject

class Constraint(ABC):
    
    """
    Ellenőrzi, hogy a paraméterben kapott adat megfelel-e a korlátorásoknak.
    Return: Igen -> True | Nem -> False
    """
    @abstractmethod
    def check(self, data: DataObject) -> bool:
        pass

    """
    Leírja, hogy a megfelelő adatnak milyen tulajdonságokkal kell rendelkeznie.
    """
    @property
    def description(self) -> str:
        return self.__class__.__name__