import numpy as np


def AWGNChannel(tx, SNR, R, random):
    """
        Input: tx - Transmitted codeword, SNR - dB, R - Code rate, use_llr - Return llr
        Output: rx - Codeword with AWGN noise
    """
    [row, col] = tx.shape

    sigma = np.sqrt(0.5 * ((10 ** ((SNR + 10 * np.log10(R)) / 10)) ** (-1)))

    rx = tx + sigma * random.normal(0.0, 1.0, (row, col))

    return 2 * rx / (sigma ** 2)
