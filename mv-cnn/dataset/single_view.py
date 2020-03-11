"""
TODO
"""

import json
import pathlib
import pandas as pd
from skimage import io, color
import torch
from torch.utils.data import Dataset

class SingleView(Dataset):
    """
    Single view dataset
    """
    def __init__(self,
                 data_dir: pathlib.PosixPath,
                 split: str,
                 metadata: str = 'metadata.csv',
                 label_map: str = 'label_map.json',
                 shuffle: bool = False):

        self.metadata = pd.read_csv(data_dir / metadata)
        self.metadata = self.metadata[self.metadata.split == split]
        self.label_map = json.load(open(data_dir / label_map, 'r'))
        self.data_dir = data_dir / split

        if shuffle or split == 'train':
            self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        path = self.data_dir / (sample.fname + '.png')
        img = io.imread(path)
        img = color.rgba2rgb(img)
        lbl = self.label_map[sample.category]

        return {'image': img, 'label': lbl}
