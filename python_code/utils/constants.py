from enum import Enum

HALF = 0.5
CLIPPING_VAL = 20
TANNER_GRAPH_CYCLE_REDUCTION = True
MAX_SIZE = 1000


class Phase(Enum):
    TRAIN = 'train'
    VAL = 'val'


class ChannelModes(Enum):
    SISO = 'SISO'


class ChannelModels(Enum):
    AWGN = 'AWGN'


class DecoderType(Enum):
    bp = 'wbp'
    wbp = 'bp'
    seq_wbp = 'seq_wbp'
    model_based_bayesian_wbp = 'model_based_bayesian_wbp'


class ModulationType(Enum):
    BPSK = 'BPSK'
