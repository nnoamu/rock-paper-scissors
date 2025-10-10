"""
Grayscale konverzió modul.
Színes képet szürkeskálássá alakít, majd visszakonvertálja BGR-re a konzisztencia miatt.
"""

import cv2
import numpy as np
from core.base_processor import PreprocessingModule


class GrayscaleConverter(PreprocessingModule):

    def __init__(self):
        super().__init__(name="Grayscale")

    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image