"""
Identity preprocessor - does nothing, just passes through the image.
Used as a placeholder when no preprocessing is needed.
"""

from core.base_processor import PreprocessingModule
from core import DataObject


class IdentityPreprocessor(PreprocessingModule):

    def __init__(self):
        super().__init__(name="Identity")

    def _process(self, input: DataObject) -> DataObject:
        return DataObject(input.data.copy())
