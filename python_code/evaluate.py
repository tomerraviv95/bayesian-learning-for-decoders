import os

from python_code import conf
from python_code.decoders.bayesian_wbp.bayesian_wbp_decoder import BayesianWBPDecoder
from python_code.decoders.wbp.wbp_decoder import WBPDecoder
from python_code.utils.constants import DecoderType

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHANNEL_TYPE_TO_TRAINER_DICT = {DecoderType.wbp.name: WBPDecoder,
                                DecoderType.model_based_bayesian_wbp.name: BayesianWBPDecoder}

if __name__ == '__main__':
    trainer = CHANNEL_TYPE_TO_TRAINER_DICT[conf.decoder_type]()
    print(trainer)
    trainer.evaluate()
