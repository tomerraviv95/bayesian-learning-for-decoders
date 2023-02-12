import numpy as np
import torch

from python_code import DEVICE
from python_code.channel.ecc_channel import load_code_parameters
from python_code.decoders.wbp.bp_nn import InputLayer, EvenLayer, OddLayer, OutputLayer
from python_code.decoders.trainer import Trainer


def llr2bits(llr_vector):
    return torch.round(torch.sigmoid(-llr_vector)).double()


def syndrome_condition(unsatisfied, llr_words, code_parityCheckMatrix):
    words = llr2bits(llr_words).float()
    syndrome = torch.fmod(torch.mm(words,
                                   torch.tensor(code_parityCheckMatrix.T).float().to(device=DEVICE)), 2)
    equal_flag = ~torch.eq(torch.sum(torch.abs(syndrome), dim=1), torch.FloatTensor(1).fill_(0).to(device=DEVICE))
    new_unsatisfied = unsatisfied[equal_flag]
    return new_unsatisfied


EPOCHS = 200


class WBPDecoder(Trainer):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.clip_tanh = 20
        self.iteration_num = 5
        self._bits_num = 63
        self._parity_bits_num = 36
        self._ecc_mat_path = r'C:/Projects/data-driven-ensembles/ECC_MATRIX'
        self.tanner_graph_cycle_reduction = True
        self.code_pcm, self.code_gm, self.t = load_code_parameters(self._bits_num, self._parity_bits_num,
                                                                   self._ecc_mat_path,
                                                                   self.tanner_graph_cycle_reduction)
        self.neurons = int(np.sum(self.code_pcm))
        self.input_layer = InputLayer(input_output_layer_size=self._bits_num, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=self.clip_tanh,
                                      bits_num=self._bits_num)
        self.even_layer = EvenLayer(self.clip_tanh, self.neurons, self.code_pcm)
        self.odd_layer = OddLayer(clip_tanh=self.clip_tanh,
                                  input_output_layer_size=self._bits_num,
                                  neurons=self.neurons,
                                  code_pcm=self.code_pcm)
        self.multiloss_output_layer = OutputLayer(neurons=self.neurons,
                                                  input_output_layer_size=self._bits_num,
                                                  code_pcm=self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons,
                                        input_output_layer_size=self._bits_num,
                                        code_pcm=self.code_pcm)
        self.odd_llr_mask_only = True
        self.even_mask_only = True
        self.multiloss_output_mask_only = True
        self.filter_in_iterations_eval = True
        self.output_mask_only = False
        self.multi_loss_flag = True

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
        self.deep_learning_setup(self.lr)
        BATCH_SIZE = 64
        for _ in range(EPOCHS):
            # select 5 samples randomly
            idx = torch.randperm(tx.shape[0])[:BATCH_SIZE]
            cur_tx, cur_rx = tx[idx], rx[idx]
            output_list, not_satisfied_list = self.forward(cur_rx)

            # calculate loss
            loss = self.calc_loss(decision=output_list[-self.iteration_num:], labels=cur_tx,
                                  not_satisfied_list=not_satisfied_list)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, x):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        x = x.float()
        output_list = [0] * (self.iteration_num + 1)
        not_satisfied_list = [0] * (self.iteration_num - 1)
        not_satisfied = torch.arange(x.size(0), dtype=torch.long, device=DEVICE)
        output_list[-1] = torch.zeros_like(x)

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = torch.index_select(x, 0, not_satisfied) + self.multiloss_output_layer.forward(
            even_output[not_satisfied], mask_only=self.multiloss_output_mask_only)

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
            output_not_satisfied = torch.index_select(x, 0, not_satisfied) + self.multiloss_output_layer.forward(
                even_output[not_satisfied], mask_only=self.multiloss_output_mask_only)
            output_list[i + 1] = output_not_satisfied.clone()
            not_satisfied_list[i] = not_satisfied.clone()

            if self.filter_in_iterations_eval and not output_not_satisfied.requires_grad:
                output_list[-1][not_satisfied] = output_not_satisfied.clone()
                not_satisfied = syndrome_condition(not_satisfied, output_not_satisfied, self.code_pcm)
            if not_satisfied.size(0) == 0:
                break
        output_list[-1][not_satisfied] = x[not_satisfied] + self.output_layer.forward(even_output[not_satisfied],
                                                                                      mask_only=self.output_mask_only)
        return output_list, not_satisfied_list
