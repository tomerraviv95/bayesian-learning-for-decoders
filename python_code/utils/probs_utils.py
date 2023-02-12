import torch

from python_code.utils.constants import HALF


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    """
    return torch.sign(p - HALF)
