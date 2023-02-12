import os
from typing import Tuple

import numpy as np
from numpy.random import default_rng

from python_code import conf
from python_code.channel.modulator import BPSKModulator


def load_code_parameters(bits_num, parity_bits_num, ecc_mat_path, tanner_graph_cycle_reduction):
    ecc_path = os.path.join(ecc_mat_path, '_'.join(['BCH', str(bits_num), str(parity_bits_num)]))
    if os.path.isfile(ecc_path + '_PCM.npy'):
        code_pcm = np.load(ecc_path + '_PCM.npy').astype(np.float32)
        code_gm = np.load(ecc_path + '_GM.npy').astype(np.float32)
    else:
        raise Exception('Code ({},{}) matrices are not exist!!!'.format(bits_num, parity_bits_num))
    if tanner_graph_cycle_reduction:
        code_pcm = (np.load(ecc_path + '_PCM_CR.npy')).astype(np.float32)
    if bits_num == 31:
        t = 1
    elif bits_num == 63:
        t = 3 if parity_bits_num == 45 else 5
    elif bits_num == 127:
        t = 10 if parity_bits_num == 64 else 5
    else:
        raise Exception('Not implemented this code type')
    return code_pcm, code_gm, t


def AWGN(tx, SNR, R, random, use_llr=True):
    """
        Input: tx - Transmitted codeword, SNR - dB, R - Code rate, use_llr - Return llr
        Output: rx - Codeword with AWGN noise
    """
    [row, col] = tx.shape

    sigma = np.sqrt(0.5 * ((10 ** ((SNR + 10 * np.log10(R)) / 10)) ** (-1)))

    rx = tx + sigma * random.normal(0.0, 1.0, (row, col))

    if use_llr:
        return 2 * rx / (sigma ** 2)
    else:
        return rx


class ECCchannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self._bits_num = 63
        self._parity_bits_num = 36
        self._ecc_mat_path = r'C:/Projects/data-driven-ensembles/ECC_MATRIX'
        self.tanner_graph_cycle_reduction = True
        self.code_pcm, self.code_gm, self.t = load_code_parameters(self._bits_num, self._parity_bits_num,
                                                                   self._ecc_mat_path,
                                                                   self.tanner_graph_cycle_reduction)
        self.encoding = lambda u: (np.dot(u, self.code_gm) % 2)
        self.modulater = BPSKModulator
        self.rate = float(self._parity_bits_num / self._bits_num)
        self.use_llr = True
        self.channel = AWGN

    def _transmit(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # generate word
        tx_pilots = self._bits_generator.integers(0, 2, size=(self._pilots_length, self._parity_bits_num))
        tx_data = self._bits_generator.integers(0, 2,
                                                size=(self._block_length - self._pilots_length, self._parity_bits_num))
        tx = np.concatenate([tx_pilots, tx_data])
        # encoding
        x = self.encoding(tx)
        # modulation
        s = self.modulater.modulate(x)
        # modulation
        # s = MODULATION_DICT[conf.modulation_type].modulate(tx.T)
        # pass through channel
        # add channel noise
        rx = self.channel(tx=s, SNR=snr, R=self.rate, use_llr=self.use_llr, random=np.random.RandomState(conf.seed))
        return x, rx

    def get_vectors(self, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # get channel values
        tx, rx = self._transmit(snr)
        return tx, rx
