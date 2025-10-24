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
        return super().description+": Batch data is not allowed. The is_batch property must be False."