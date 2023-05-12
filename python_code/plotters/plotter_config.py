from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DecoderType


class PlotType(Enum):
    ## The Three Figures for the Paper
    BY_SNR_63_45_CODE = 'BY_SNR_63_45_CODE'
    BY_SNR_127_64_CODE = 'BY_SNR_127_64_CODE'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str]:
    if plot_type in [PlotType.BY_SNR_63_45_CODE, PlotType.BY_SNR_127_64_CODE]:
        params_dicts = [
            {'val_snr': 4, 'detector_type': DecoderType.bp.name},
            {'val_snr': 5, 'detector_type': DecoderType.bp.name},
            {'val_snr': 6, 'detector_type': DecoderType.bp.name},
            {'val_snr': 7, 'detector_type': DecoderType.bp.name},
            {'val_snr': 4, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 5, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 6, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 7, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 4, 'detector_type': DecoderType.bayesian_wbp.name, 'train_blocks_num': 5},
            {'val_snr': 5, 'detector_type': DecoderType.bayesian_wbp.name, 'train_blocks_num': 5},
            {'val_snr': 6, 'detector_type': DecoderType.bayesian_wbp.name, 'train_blocks_num': 5},
            {'val_snr': 7, 'detector_type': DecoderType.bayesian_wbp.name, 'train_blocks_num': 5},
            {'val_snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 5},
            {'val_snr': 5, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 5},
            {'val_snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 5},
            {'val_snr': 7, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 5},
        ]
        values = list(range(4, 8, 1))
        xlabel, ylabel = 'SNR', 'SER'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, values, xlabel, ylabel
