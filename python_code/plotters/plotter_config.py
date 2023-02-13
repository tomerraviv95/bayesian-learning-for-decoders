from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import DecoderType


class PlotType(Enum):
    ## The Three Figures for the Paper
    BY_SNR_LONG_CODE = 'BY_SNR_LONG_CODE'
    BY_SNR_SHORT_CODE = 'BY_SNR_SHORT_CODE'


def get_config(plot_type: PlotType) -> Tuple[List[Dict], list, str, str]:
    if plot_type == PlotType.BY_SNR_LONG_CODE:
        params_dicts = [
            {'snr': 2, 'detector_type': DecoderType.bp.name},
            {'snr': 4, 'detector_type': DecoderType.bp.name},
            {'snr': 6, 'detector_type': DecoderType.bp.name},
            {'snr': 8, 'detector_type': DecoderType.bp.name},
            {'snr': 2, 'detector_type': DecoderType.wbp.name},
            {'snr': 4, 'detector_type': DecoderType.wbp.name},
            {'snr': 6, 'detector_type': DecoderType.wbp.name},
            {'snr': 8, 'detector_type': DecoderType.wbp.name},
            {'snr': 2, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
        ]
        values = list(range(2, 9, 2))
        xlabel, ylabel = 'SNR', 'SER'
    elif plot_type == PlotType.BY_SNR_SHORT_CODE:
        params_dicts = [
            {'snr': 2, 'detector_type': DecoderType.bp.name},
            {'snr': 4, 'detector_type': DecoderType.bp.name},
            {'snr': 6, 'detector_type': DecoderType.bp.name},
            {'snr': 8, 'detector_type': DecoderType.bp.name},
            {'snr': 2, 'detector_type': DecoderType.wbp.name},
            {'snr': 4, 'detector_type': DecoderType.wbp.name},
            {'snr': 6, 'detector_type': DecoderType.wbp.name},
            {'snr': 8, 'detector_type': DecoderType.wbp.name},
            {'snr': 2, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'snr': 4, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'snr': 6, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
            {'snr': 8, 'detector_type': DecoderType.model_based_bayesian_wbp.name},
        ]
        values = list(range(2, 9, 2))
        xlabel, ylabel = 'SNR', 'SER'
    else:
        raise ValueError('No such plot type!!!')

    return params_dicts, values, xlabel, ylabel
