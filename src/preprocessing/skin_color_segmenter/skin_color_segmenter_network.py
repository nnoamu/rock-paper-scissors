"""
Bőrszín eldöntő neuronháló.
Bemenet: 3 elemű [0, 1] float RGB vektor (RGB pixel).
Kimenet: [0, 1] valószínűség, hogy a pixel bőrpixel.
"""

from torch import nn

class SkinColorSegmenterNetwork(nn.Module):
    def __init__(self, num_layers: int=4, layer_density: int=10, activation_function=nn.ReLU):
        super().__init__()
        layers=[]
        layers.append(nn.Linear(in_features=3, out_features=layer_density))
        layers.append(activation_function())

        for _ in range(1, num_layers, 1):
            layers.append(nn.Linear(in_features=layer_density, out_features=layer_density))
            layers.append(activation_function())

        layers.append(nn.Linear(in_features=layer_density, out_features=1))
        layers.append(nn.Sigmoid())
        
        self.network=nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
        