"""
Előfeldolgozó modulok ősosztálya.
Tartalmazott korlátozások:
- ImageLikeConstraint

Példa használat: Gauss-szűrés 1 képre
    class MyPreprocessor(PreprocessingModule):
        def _process(self, input: DataObject) -> DataObject:
            return Dataobject(cv2.GaussianBlur(input.data, (5, 5), 0))
"""

from .base_module import BaseModule
from constraints import ImageLikeConstraint
from typing import Any, Dict, List, final

from .data_object import DataObject


class PreprocessingModule(BaseModule):

    def __init__(self, name: str):
        super().__init__(name)
        self.add_constraint(ImageLikeConstraint())

        self.params: Dict[str, Any] = {}
    
    @final
    def preprocess(self, image: DataObject | List[DataObject]) -> DataObject | List[DataObject]:
        return self.process(image)

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value