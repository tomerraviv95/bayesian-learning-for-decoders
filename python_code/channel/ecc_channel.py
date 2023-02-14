from typing import Tuple

import numpy as np
from numpy.random import default_rng

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.channel.awgn_channel import AWGNChannel
from python_code.channel.modulator import BPSKModulation
from python_code.utils.constants import TANNER_GRAPH_CYCLE_REDUCTION
from python_code.utils.python_utils import load_code_parameters


class EccChannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self._code_bits = conf.code_bits
        self._info_bits = conf.info_bits
        self.code_pcm, self.code_gm = load_code_parameters(self._code_bits, self._info_bits,
                                                           ECC_MATRICES_DIR, TANNER_GRAPH_CYCLE_REDUCTION)
        self.encoding = lambda u: (np.dot(u, self.code_gm) % 2)
        self.modulation = BPSKModulation
        self.channel = AWGNChannel
        self.rate = float(self._info_bits / self._code_bits)

    def _transmit(self, val_snr: float) -> Tuple[np.ndarray, np.ndarray]:
        rx_pilots, x_pilots = [], []
        snrs_num = conf.train_snr_end - conf.train_snr_start + 1
        for train_snr in range(conf.train_snr_start, conf.train_snr_end + 1):
            cur_rx_pilots, cur_x_pilots = self._transmit_single(train_snr, self._pilots_length // snrs_num)
            rx_pilots.append(cur_rx_pilots)
            x_pilots.append(cur_x_pilots)
        rx_pilots, x_pilots = np.concatenate(rx_pilots), np.concatenate(x_pilots)
        rx_data, x_data = self._transmit_single(val_snr, self._block_length - self._pilots_length)
        x = np.concatenate([x_pilots, x_data])
        rx = np.concatenate([rx_pilots, rx_data])
        return x, rx

    def _transmit_single(self, snr, size):
        tx = self._bits_generator.integers(0, 2, size=(size, self._info_bits))
        x = self.encoding(tx)
        s = self.modulation.modulate(x)
        rx = self.channel(tx=s, SNR=snr, R=self.rate, random=np.random.RandomState(conf.seed))
        return rx, x

    def get_vectors(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # get channel values
        tx, rx = self._transmit(snr)
        return tx, rx
