"""
Kötegelt adatot tiltó korlátozás.
"""

from core.data_object import DataObject
from core.base_constraint import Constraint

class NoBatchConstraint(Constraint):

    def check(self, data: DataObject) -> bool:
        return not data.is_batch
    
    @property
    def description(self) -> str:
        return super().description+": Az adat nem lehet kötegelt. Az is_batch adattag False kell legyen."