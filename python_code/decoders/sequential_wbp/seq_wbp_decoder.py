import torch

from python_code.decoders.bp_nn import InputLayer, OddLayer, EvenLayer, OutputLayer
from python_code.decoders.trainer import Trainer
from python_code.utils.constants import CLIPPING_VAL, Phase

EPOCHS = 500
BATCH_SIZE = 64


class SequentialWBPDecoder(Trainer):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.output_mask_only = True
        self.initialize_layers()
        self.deep_learning_setup(self.lr)

    def __str__(self):
        return 'Sequential WBP Decoder'

    def initialize_layers(self):
        self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL,
                                      bits_num=self._code_bits)
        self.odd_layer = OddLayer(clip_tanh=CLIPPING_VAL,
                                  input_output_layer_size=self._code_bits,
                                  neurons=self.neurons,
                                  code_pcm=self.code_pcm)
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons,
                                        input_output_layer_size=self._code_bits,
                                        code_pcm=self.code_pcm)

    def calc_loss(self, cur_tx, output):
        # calculate loss
        loss = self.criterion(input=-output, target=cur_tx)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        for e in range(EPOCHS):
            # select samples randomly
            idx = torch.randperm(tx.shape[0])[:BATCH_SIZE]
            cur_tx, cur_rx = tx[idx], rx[idx]
            # initialize
            even_output = self.input_layer.forward(cur_rx)
            for i in range(0, self.iteration_num - 1):
                # odd - variables to check
                odd_output = self.odd_layer.forward(even_output, cur_rx, llr_mask_only=self.odd_llr_mask_only)
                # even - check to variables
                even_output_cur = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                # output layer
                output = cur_rx + self.output_layer.forward(even_output_cur, mask_only=self.output_mask_only)
                self.calc_loss(cur_tx, output)
                with torch.no_grad():
                    # update the even_output to the next layer
                    # odd - variables to check
                    odd_output = self.odd_layer.forward(even_output_cur, cur_rx, llr_mask_only=self.odd_llr_mask_only)
                    # even - check to variables
                    even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)

    def forward(self, x, phase: Phase):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        output_list = [0] * self.iteration_num

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = x + self.output_layer.forward(even_output, mask_only=self.output_layer)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(0, self.iteration_num - 1):
            # odd - variables to check
            odd_output = self.odd_layer.forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
            # output layer
            output = x + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
            output_list[i + 1] = output.clone()

        return output_list
