from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DecoderType


class PlotType(Enum):
    ## The Three Figures for the Paper
    BY_SNR_LONG_CODE = 'BY_SNR_LONG_CODE'
    BY_SNR_LONG_CODE_10_ITERS = 'BY_SNR_LONG_CODE_10_ITERS'
    BY_SNR_LONG_CODE_20_ITERS = 'BY_SNR_LONG_CODE_20_ITERS'
    BY_SNR_SHORT_CODE = 'BY_SNR_SHORT_CODE'
    BY_SNR_SHORT_CODE_10_ITERS = 'BY_SNR_SHORT_CODE_10_ITERS'
    BY_SNR_SHORT_CODE_20_ITERS = 'BY_SNR_SHORT_CODE_20_ITERS'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str]:
    if plot_type == PlotType.BY_SNR_LONG_CODE:
        params_dicts = [
            {'val_snr': 4, 'detector_type': DecoderType.bp.name},
            {'val_snr': 5, 'detector_type': DecoderType.bp.name},
            {'val_snr': 6, 'detector_type': DecoderType.bp.name},
            {'val_snr': 7, 'detector_type': DecoderType.bp.name},
            {'val_snr': 8, 'detector_type': DecoderType.bp.name},
            {'val_snr': 4, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 5, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 6, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 7, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 8, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 5, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 7, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
        ]
        values = list(range(4, 9, 1))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.BY_SNR_SHORT_CODE:
        params_dicts = [
            {'val_snr': 4, 'detector_type': DecoderType.bp.name},
            {'val_snr': 5, 'detector_type': DecoderType.bp.name},
            {'val_snr': 6, 'detector_type': DecoderType.bp.name},
            {'val_snr': 7, 'detector_type': DecoderType.bp.name},
            {'val_snr': 8, 'detector_type': DecoderType.bp.name},
            {'val_snr': 4, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 5, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 6, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 7, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 8, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 5, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 7, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
        ]
        values = list(range(4, 9, 1))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.BY_SNR_LONG_CODE_10_ITERS:
        params_dicts = [
            {'val_snr': 4, 'detector_type': DecoderType.bp.name},
            {'val_snr': 5, 'detector_type': DecoderType.bp.name},
            {'val_snr': 6, 'detector_type': DecoderType.bp.name},
            {'val_snr': 7, 'detector_type': DecoderType.bp.name},
            {'val_snr': 8, 'detector_type': DecoderType.bp.name},
            {'val_snr': 4, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 5, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 6, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 7, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 8, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 5, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 7, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
        ]
        values = list(range(4, 9, 1))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.BY_SNR_SHORT_CODE_10_ITERS:
        params_dicts = [
            {'val_snr': 4, 'detector_type': DecoderType.bp.name},
            {'val_snr': 5, 'detector_type': DecoderType.bp.name},
            {'val_snr': 6, 'detector_type': DecoderType.bp.name},
            {'val_snr': 7, 'detector_type': DecoderType.bp.name},
            {'val_snr': 8, 'detector_type': DecoderType.bp.name},
            {'val_snr': 4, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 5, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 6, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 7, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 8, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 5, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 7, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
        ]
        values = list(range(4, 9, 1))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.BY_SNR_LONG_CODE_20_ITERS:
        params_dicts = [
            {'val_snr': 4, 'detector_type': DecoderType.bp.name},
            {'val_snr': 5, 'detector_type': DecoderType.bp.name},
            {'val_snr': 6, 'detector_type': DecoderType.bp.name},
            {'val_snr': 7, 'detector_type': DecoderType.bp.name},
            {'val_snr': 8, 'detector_type': DecoderType.bp.name},
            {'val_snr': 4, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 5, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 6, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 7, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 8, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 5, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 7, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'val_snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
        ]
        values = list(range(4, 9, 1))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.BY_SNR_SHORT_CODE_20_ITERS:
        params_dicts = [
            {'val_snr': 4, 'detector_type': DecoderType.bp.name},
            {'val_snr': 5, 'detector_type': DecoderType.bp.name},
            {'val_snr': 6, 'detector_type': DecoderType.bp.name},
            {'val_snr': 7, 'detector_type': DecoderType.bp.name},
            {'val_snr': 8, 'detector_type': DecoderType.bp.name},
            {'val_snr': 4, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 5, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 6, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 7, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 8, 'detector_type': DecoderType.wbp.name, 'train_blocks_num': 30},
            {'val_snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 5, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 7, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
            {'val_snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name, 'train_blocks_num': 3},
        ]
        values = list(range(4, 9, 1))
        xlabel, ylabel = 'SNR', 'SER'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, values, xlabel, ylabel
