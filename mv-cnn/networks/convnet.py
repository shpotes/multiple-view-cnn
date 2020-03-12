from typing import List, Tuple, Union

import torch.nn as nn

def parseConvNet(arch: List[Union[Tuple[int, int, int]], int]) -> nn.Module:
    model = []
    for i in range(len(arch) - 1):
        if isinstance(arch[i], list):
            model.append(nn.Conv3d(*arch[i]))

            if isinstance(arch[i + 1], int):
                model.append(nn.Flatten())


        elif isinstance(arch[i], int):
            model.append(nn.Linear(arch[i]))

        model.append(nn.ReLU())

    return nn.Sequential(*model)
