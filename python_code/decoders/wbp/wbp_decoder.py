import torch

from python_code import DEVICE
from python_code.decoders.trainer import Trainer
from python_code.decoders.bp.bp_nn import InputLayer, OddLayer, EvenLayer, OutputLayer
from python_code.utils.constants import MAX_SIZE, EPOCHS, BATCH_SIZE, CLIPPING_VAL
from python_code.utils.python_utils import syndrome_condition


class WBPDecoder(Trainer):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.is_online_training = True

    def __str__(self):
        return 'WBP Decoder'

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


    def calc_loss(self, decision, labels, not_satisfied_list):
        loss = self.criterion(input=-decision[-1], target=labels)
        if self.multi_loss_flag:
            for iteration in range(self.iteration_num - 1):
                if type(not_satisfied_list[iteration]) == int:
                    break
                current_loss = self.criterion(input=-decision[iteration],
                                              target=torch.index_select(labels, 0, not_satisfied_list[iteration]))
                loss += current_loss
        return loss

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        self.initialize_layers()
        self.deep_learning_setup(self.lr)
        for _ in range(EPOCHS):
            # select 5 samples randomly
            idx = torch.randperm(tx.shape[0])[:BATCH_SIZE]
            cur_tx, cur_rx = tx[idx], rx[idx]
            output_list, not_satisfied_list = self._forward(cur_rx)
            # calculate loss
            loss = self.calc_loss(decision=output_list[-self.iteration_num:], labels=cur_tx,
                                  not_satisfied_list=not_satisfied_list)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _forward(self, x):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        output_list = [0] * self.iteration_num
        not_satisfied_list = [0] * (self.iteration_num - 1)
        not_satisfied = torch.arange(x.size(0), dtype=torch.long, device=DEVICE)

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = torch.index_select(x, 0, not_satisfied) + self.output_layer.forward(
            even_output[not_satisfied], mask_only=self.output_layer)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(0, self.iteration_num - 1):
            # odd - variables to check
            odd_output_not_satisfied = self.odd_layer.forward(torch.index_select(even_output, 0, not_satisfied),
                                                              torch.index_select(x, 0, not_satisfied),
                                                              llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output[not_satisfied] = self.even_layer.forward(odd_output_not_satisfied,
                                                                 mask_only=self.even_mask_only)
            # output layer
            output_not_satisfied = torch.index_select(x, 0, not_satisfied) + self.output_layer.forward(
                even_output[not_satisfied], mask_only=self.output_mask_only)
            output_list[i + 1] = output_not_satisfied.clone()
            not_satisfied_list[i] = not_satisfied.clone()

            if self.filter_in_iterations_eval and not output_not_satisfied.requires_grad:
                output_list[-1][not_satisfied] = output_not_satisfied.clone()
                not_satisfied = syndrome_condition(not_satisfied, output_not_satisfied, self.code_pcm)
            if not_satisfied.size(0) == 0:
                break
        return output_list, not_satisfied_list

    def forward(self, x):
        batch_size = min(MAX_SIZE, x.shape[0])
        total_decoded_words = []
        for i in range(x.shape[0] // batch_size):
            output_list, not_satisfied_list = self._forward(x[i * batch_size:(i + 1) * batch_size])
            decoded_words = torch.round(torch.sigmoid(-output_list[-1]))
            total_decoded_words.append(decoded_words)
        total_decoded_words = torch.cat(total_decoded_words)
        return total_decoded_words
