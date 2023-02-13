from enum import Enum

HALF = 0.5
CLIPPING_VAL = 20
TANNER_GRAPH_CYCLE_REDUCTION = True

class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ChannelModes(Enum):
    SISO = 'SISO'


class ChannelModels(Enum):
    AWGN = 'AWGN'


class DecoderType(Enum):
    wbp = 'wbp'
    seq_wbp = 'seq_wbp'
    model_based_bayesian_wbp = 'model_based_bayesian_wbp'


class ModulationType(Enum):
    BPSK = 'BPSK'
