import os
import pickle as pkl
from typing import Dict, Any

import numpy as np


def load_code_parameters(bits_num, parity_bits_num, ecc_mat_path, tanner_graph_cycle_reduction):
    ecc_path = os.path.join(ecc_mat_path, '_'.join(['BCH', str(bits_num), str(parity_bits_num)]))
    if os.path.isfile(ecc_path + '_PCM.npy'):
        code_pcm = np.load(ecc_path + '_PCM.npy').astype(np.float32)
        code_gm = np.load(ecc_path + '_GM.npy').astype(np.float32)
    else:
        raise Exception('Code ({},{}) matrices are not exist!!!'.format(bits_num, parity_bits_num))
    if tanner_graph_cycle_reduction:
        code_pcm = (np.load(ecc_path + '_PCM_CR.npy')).astype(np.float32)
    return code_pcm, code_gm


def save_pkl(pkls_path: str, array: np.ndarray, type: str):
    output = open(pkls_path + '_' + type + '.pkl', 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str, type: str) -> Dict[Any, Any]:
    output = open(pkls_path + '_' + type + '.pkl', 'rb')
    return pkl.load(output)
