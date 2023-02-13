import os
import pickle as pkl
from typing import Dict, Any

import numpy as np
import torch

from python_code import DEVICE


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


def llr2bits(llr_vector):
    return torch.round(torch.sigmoid(-llr_vector))


def syndrome_condition(unsatisfied, llr_words, code_pcm):
    words = llr2bits(llr_words).float()
    syndrome = torch.fmod(torch.mm(words, torch.tensor(code_pcm.T).float().to(device=DEVICE)), 2)
    equal_flag = ~torch.eq(torch.sum(torch.abs(syndrome), dim=1), torch.FloatTensor(1).fill_(0).to(device=DEVICE))
    new_unsatisfied = unsatisfied[equal_flag]
    return new_unsatisfied


def save_pkl(pkls_path: str, array: np.ndarray, type: str):
    output = open(pkls_path + '_' + type + '.pkl', 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str, type: str) -> Dict[Any, Any]:
    output = open(pkls_path + '_' + type + '.pkl', 'rb')
    return pkl.load(output)
