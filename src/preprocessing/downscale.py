"""
Downscale preprocessor - resizes image to smaller resolution for faster processing.
"""

import cv2
import numpy as np
from core.base_processor import PreprocessingModule
from core import DataObject


class DownscalePreprocessor(PreprocessingModule):
    """
    Downscales images to a maximum dimension while preserving aspect ratio.

    Args:
        max_size: Maximum width or height in pixels (default 640)
        interpolation: OpenCV interpolation method (default INTER_AREA for downscaling)
    """

    def __init__(self, max_size: int = 640, interpolation: int = cv2.INTER_AREA):
        super().__init__(name=f"Downscale({max_size})")
        self.max_size = max_size
        self.interpolation = interpolation

    def _process(self, input: DataObject) -> DataObject:
        image = input.data

        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]

        # Check if downscaling is needed
        if max(h, w) <= self.max_size:
            return DataObject(image.copy())

        # Calculate new dimensions preserving aspect ratio
        if w > h:
            new_w = self.max_size
            new_h = int(h * (self.max_size / w))
        else:
            new_h = self.max_size
            new_w = int(w * (self.max_size / h))

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)

        return DataObject(resized)
