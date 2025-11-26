from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SkinColorDataset(Dataset):
    def __init__(self, file_path):
        self.data=pd.read_csv(file_path, sep=',', dtype=np.uint8)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        color=self.data.loc[idx, ['R', 'G', 'B']].to_numpy(dtype=np.float32)/255
        label=self.data.loc[idx, 'y']
        label=np.float32([label])
        return color, label