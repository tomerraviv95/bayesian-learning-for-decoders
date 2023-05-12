import os

from python_code import conf
from python_code.decoders.bayesian_wbp.bayesian_wbp_decoder import BayesianWBPDecoder
from python_code.decoders.bp.bp_decoder import BPDecoder
from python_code.decoders.model_based_bayesian_wbp.model_based_bayesian_wbp_decoder import ModelBasedBayesianWBPDecoder
from python_code.decoders.sequential_wbp.seq_wbp_decoder import SequentialWBPDecoder
from python_code.decoders.wbp.wbp_decoder import WBPDecoder
from python_code.utils.constants import DecoderType

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHANNEL_TYPE_TO_TRAINER_DICT = {DecoderType.bp.name: BPDecoder,
                                DecoderType.wbp.name: WBPDecoder,
                                DecoderType.seq_wbp.name: SequentialWBPDecoder,
                                DecoderType.bayesian_wbp.name: BayesianWBPDecoder,
                                DecoderType.model_based_bayesian_wbp.name: ModelBasedBayesianWBPDecoder}

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.decoder_type]()
    print(trainer)
    trainer.train_and_eval()
