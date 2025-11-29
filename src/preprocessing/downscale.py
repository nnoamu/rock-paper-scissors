"""
Grayscale konverzió modul.
Színes képet szürkeskálássá alakít. 1 csatornás képet ad vissza.
(Base Modulból származtatott modul minta implementáció.)
"""

import cv2
from core import PreprocessingModule, DataObject


class DownscaleModule(PreprocessingModule):

    def __init__(self, max_dimension_length=300):
        super().__init__(name="Downscale")
        self.max_image_size=max_dimension_length

    def _process(self, input: DataObject) -> DataObject:
        img=input.data
        shape=input.shape[:2]
        mx=max(shape)

        if mx>self.max_image_size:
            tg=self.max_image_size
            r=tg/mx
            new_w=round(shape[0]*r)
            new_h=round(shape[1]*r)
            img=cv2.resize(img, (new_w, new_h))

        return DataObject(img)