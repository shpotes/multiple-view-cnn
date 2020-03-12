"""
TODO
"""

import torch

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Return mini-batch accuracy
    """
    return torch.sum(output.max(dim=1)[1] == target)
