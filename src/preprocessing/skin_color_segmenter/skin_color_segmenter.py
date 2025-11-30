"""
Bőrszín alapú szegmentáló modul.
RGB képet szürkeárnyalatos uint8 képpé alakít. A pixelek értékei jelölik, hogy az adott helyen milyen valószínűséggel bőrszínű a pixel (0 -> 0%, 255 -> 100).
"""

import torch
from configparser import ConfigParser
from pathlib import Path
import numpy as np
from core.base_processor import PreprocessingModule
from core import DataObject
from constraints import TypeConstraint, RGBImageConstraint
from .skin_color_segmenter_network import SkinColorSegmenterNetwork

class SkinColorSegmenterModule(PreprocessingModule):

    def __init__(self, model_path: str):
        super().__init__(name="SkinColorSegmenter")
        self.add_constraint(TypeConstraint(np.uint8))
        self.add_constraint(RGBImageConstraint())
        
        config=ConfigParser()
        config_file=Path(model_path).joinpath('config.txt')
        if len(config.read([config_file]))==0:
            print("OOF")
            raise Exception("model config file not found")
        
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=SkinColorSegmenterNetwork(num_layers=int(config["NETWORK"]["num_layers"]), layer_density=int(config["NETWORK"]["layer_density"]))
        self.model.load_state_dict(torch.load(Path(model_path).joinpath('model.pth'), map_location=self.device, weights_only=True), assign=True)
        self.model.to(self.device)
        self.model.eval()


    def _process(self, input: DataObject) -> DataObject:
        img=input.data
        img=torch.tensor(input.data/255, dtype=torch.float32)
        img=img.to(self.device)
        height, width, channels=img.shape
        img=img.reshape((height*width, channels))
        
        with torch.no_grad():
            img=self.model(img)
        
        img=img.reshape((height, width)).cpu()
        img=np.array(img*255, dtype=np.uint8)

        return DataObject(img)

