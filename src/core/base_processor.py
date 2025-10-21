"""
Előfeldolgozó modulok ősosztálya.

Példa használat:
    class MyPreprocessor(PreprocessingModule):
        def process(self, image: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(image, (5, 5), 0)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class PreprocessingModule(ABC):

    def __init__(self, name: str):
        self.name = name
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def __str__(self) -> str:
        return self.name