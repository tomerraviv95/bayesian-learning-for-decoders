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
    def __init__(self, block_size: int):
        self._block_length = block_size
        self._bits_generator = default_rng(seed=conf.seed)
        self._code_bits = conf.code_bits
        self._info_bits = conf.info_bits
        self.code_pcm, self.code_gm = load_code_parameters(self._code_bits, self._info_bits,
                                                           ECC_MATRICES_DIR, TANNER_GRAPH_CYCLE_REDUCTION)
        self.encoding = lambda u: (np.dot(u, self.code_gm) % 2)
        self.modulation = BPSKModulation
        self.channel = AWGNChannel
        self.rate = float(self._info_bits / self._code_bits)

    def _transmit(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        tx = self._bits_generator.integers(0, 2, size=(self._block_length, self._info_bits))
        x = self.encoding(tx)
        s = self.modulation.modulate(x)
        rx = self.channel(tx=s, SNR=snr, R=self.rate, random=np.random.RandomState(conf.seed))
        return x, rx

    def get_vectors(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # get channel values
        tx, rx = self._transmit(snr)
        return tx, rx
