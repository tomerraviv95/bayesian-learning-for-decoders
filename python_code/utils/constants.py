from enum import Enum, auto

HALF = 0.5
CLIPPING_VAL = 10
TANNER_GRAPH_CYCLE_REDUCTION = True


class Phase(Enum):
    TRAIN = 'train'
    VAL = 'val'


class ChannelModes(Enum):
    SISO = 'SISO'


class ChannelModels(Enum):
    AWGN = 'AWGN'


class DecoderType(Enum):
    bp = 'bp'
    wbp = 'wbp'
    seq_wbp = 'seq_wbp'
    bayesian_wbp = 'bayesian_wbp'
    model_based_bayesian_wbp = 'model_based_bayesian_wbp'


class ModulationType(Enum):
    BPSK = 'BPSK'
