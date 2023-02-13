from typing import Tuple

import numpy as np
from numpy.random import default_rng

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.channel.modulator import BPSKModulator
from python_code.utils.python_utils import load_code_parameters


def AWGN(tx, SNR, R, random):
    """
        Input: tx - Transmitted codeword, SNR - dB, R - Code rate, use_llr - Return llr
        Output: rx - Codeword with AWGN noise
    """
    [row, col] = tx.shape

    sigma = np.sqrt(0.5 * ((10 ** ((SNR + 10 * np.log10(R)) / 10)) ** (-1)))

    rx = tx + sigma * random.normal(0.0, 1.0, (row, col))

    return 2 * rx / (sigma ** 2)


class ECCchannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self._code_bits = conf.code_bits
        self._info_bits = conf.info_bits
        self.tanner_graph_cycle_reduction = True
        self.code_pcm, self.code_gm = load_code_parameters(self._code_bits, self._info_bits,
                                                           ECC_MATRICES_DIR, self.tanner_graph_cycle_reduction)
        self.encoding = lambda u: (np.dot(u, self.code_gm) % 2)
        self.modulater = BPSKModulator
        self.rate = float(self._info_bits / self._code_bits)
        self.channel = AWGN

    def _transmit(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # generate word
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, self._info_bits))
        tx_data = self._bits_generator.integers(0, 2, size=(self._block_length - self._pilots_length, self._info_bits))
        tx = np.concatenate([tx_pilots, tx_data])
        # encoding
        x = self.encoding(tx)
        # modulation
        s = self.modulater.modulate(x)
        # add channel noise
        rx = self.channel(tx=s, SNR=snr, R=self.rate, random=np.random.RandomState(conf.seed))
        return x, rx

    def get_vectors(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # get channel values
        tx, rx = self._transmit(snr)
        return tx, rx
