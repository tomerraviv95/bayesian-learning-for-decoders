import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from python_code import DEVICE
from python_code.channel.ecc_channel import EccChannel


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation.
    Returns (transmitted, received, channel_coefficients) batch.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.channel_type = EccChannel(block_size)

    def get_snr_data(self, snr: float, database: list):
        if database is None:
            database = []
        tx, rx = self.channel_type.get_vectors(snr)
        database.append((tx, rx))

    def __getitem__(self, snr_list: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, snr, database) for snr in snr_list]
        tx, rx = (np.concatenate(arrays) for arrays in zip(*database))
        tx, rx = torch.Tensor(tx).to(device=DEVICE), torch.from_numpy(rx).to(device=DEVICE).float()
        return tx, rx

    def __len__(self):
        return self.block_size
