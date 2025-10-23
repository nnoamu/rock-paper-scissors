"""
Általános adatreprezentáció a teljes folyamat során használt modulok közti adattovábbításra.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Final
import numpy as np

@dataclass
class DataObject:
    data: Final[np.ndarray]
    named_data: Optional[Dict[str, Any]]
    
    is_batch: Final[bool]
    dimension: Final[int]
    shape: Final[tuple]
    dtype: Final[np.dtype]
    min_val: Final[Any]
    max_val: Final[Any]
    metadata: Optional[Dict[str, Any]]

    @staticmethod
    def description() -> str:
        return "General data object. Can hold data of any type, shape etc."

    def __init__(self, data: np.ndarray, is_batch: bool, named_data: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, Any]]=None):
        self.data=data
        self.is_batch=is_batch
        self.named_data=named_data
        self.metadata=metadata

        self.shape=self.data.shape[(1 if is_batch else 0) :]
        self.dimension=len(self.shape)
        self.dtype=data.dtype
        self.min_val=np.min(self.data)
        self.max_val=np.max(self.data)

    def get_summary(self) -> str:
        return f"{self.dtype} | {self.dimension}D | {self.shape}"