from typing import List

import torch.nn as nn

def parseNet(arch: List[int], dropout_rate: float = 0.5) -> nn.Module:
    model = []
    for layer in arch:
        model.append(nn.Linear(layer))
        model.append(nn.Dropout(dropout_rate))
        model.append(nn.ReLU())

    return nn.Sequential(*model)
