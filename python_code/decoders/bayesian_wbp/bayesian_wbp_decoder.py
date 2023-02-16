import torch

from python_code import DEVICE
from python_code.decoders.bayesian_wbp.bayesian_bp_nn import BayesianOddLayer
from python_code.decoders.bp_nn import InputLayer, EvenLayer, OutputLayer
from python_code.decoders.trainer import Trainer
from python_code.utils.constants import HALF, CLIPPING_VAL, Phase
from python_code.utils.python_utils import syndrome_condition

EPOCHS = 300
BATCH_SIZE = 64


class BayesianWBPDecoder(Trainer):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.output_mask_only = True
        self.ensemble_num = 5
        self.kl_beta = 1e-4
        self.beta = 1e-2
        self.initialize_layers()
        self.deep_learning_setup(self.lr)

    def __str__(self):
        return 'Bayesian WBP Decoder'

    def initialize_layers(self):
        self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL,
                                      bits_num=self._code_bits)
        self.odd_layer = BayesianOddLayer(clip_tanh=CLIPPING_VAL,
                                          input_output_layer_size=self._code_bits,
                                          neurons=self.neurons,
                                          code_pcm=self.code_pcm)
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons,
                                        input_output_layer_size=self._code_bits,
                                        code_pcm=self.code_pcm)

    def calc_loss(self, cur_tx, output, arm_original, arm_tilde, u_list, dropout_logit, kl_term):
        # calculate loss
        loss = self.criterion(input=-output, target=cur_tx)
        # ARM Loss
        arm_loss = 0
        for ens_ind in range(self.ensemble_num):
            loss_term_arm_original = self.criterion(input=-arm_original[ens_ind], target=cur_tx)
            loss_term_arm_tilde = self.criterion(input=-arm_tilde[ens_ind], target=cur_tx)
            arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
            grad_logit = arm_delta * (u_list[ens_ind] - HALF)
            arm_loss += torch.matmul(grad_logit, dropout_logit.T)
        arm_loss = torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * kl_term
        extra_loss = arm_loss + kl_term
        loss += self.beta * extra_loss
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
                arm_original, arm_tilde, u_list, kl_term = [], [], [], 0
                output = 0
                for _ in range(self.ensemble_num):
                    # odd - variables to check
                    odd_output = self.odd_layer.forward(even_output, cur_rx, llr_mask_only=self.odd_llr_mask_only)
                    # propagate
                    output += self.propagate(cur_rx, odd_output)
                    ### ARM
                    arm_original.append(output)
                    init_arm_tilde = self.odd_layer.init_arm_tilde
                    output_tilde = self.propagate(cur_rx, init_arm_tilde)
                    arm_tilde.append(output_tilde)
                    u_list.append(self.odd_layer.u)
                kl_term = self.odd_layer.kl_term
                output /= self.ensemble_num
                self.calc_loss(cur_tx, output, arm_original, arm_tilde, u_list, self.odd_layer.dropout_logit, kl_term)
                with torch.no_grad():
                    even_output_est = 0
                    # update the even_output to the next layer
                    for _ in range(self.ensemble_num):
                        # odd - variables to check
                        odd_output = self.odd_layer.forward(even_output, cur_rx, llr_mask_only=self.odd_llr_mask_only)
                        # even - check to variables
                        even_output_est += self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                    even_output = even_output_est / self.ensemble_num

    def propagate(self, cur_rx, odd_output):
        # even - check to variables
        even_output_cur = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
        # output layer
        output = cur_rx + self.output_layer.forward(even_output_cur, mask_only=self.output_mask_only)
        return output

    def forward(self, x, phase: Phase):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        output_list = [0] * (self.iteration_num)
        not_satisfied_list = [0] * (self.iteration_num - 1)
        not_satisfied = torch.arange(x.size(0), dtype=torch.long, device=DEVICE)
        output_list[-1] = torch.zeros_like(x)

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = torch.index_select(x, 0, not_satisfied) + self.output_layer.forward(
            even_output[not_satisfied], mask_only=self.output_layer)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(0, self.iteration_num - 1):
            output_not_satisfied, total_even_output_not_satisfied = 0, 0
            for _ in range(self.ensemble_num):
                # odd - variables to check
                odd_output_not_satisfied = self.odd_layer.forward(torch.index_select(even_output, 0, not_satisfied),
                                                                  torch.index_select(x, 0, not_satisfied),
                                                                  llr_mask_only=self.odd_llr_mask_only)
                # even - check to variables
                even_output_not_satisfied = self.even_layer.forward(odd_output_not_satisfied,
                                                                    mask_only=self.even_mask_only)
                total_even_output_not_satisfied += even_output_not_satisfied
                # output layer
                output_not_satisfied += torch.index_select(x, 0, not_satisfied) + self.output_layer.forward(
                    even_output_not_satisfied, mask_only=self.output_mask_only)

            even_output[not_satisfied] = total_even_output_not_satisfied / self.ensemble_num
            output_not_satisfied /= self.ensemble_num
            output_list[i + 1] = output_not_satisfied.clone()
            not_satisfied_list[i] = not_satisfied.clone()

            if not_satisfied.size(0) == 0:
                break
        return output_list, not_satisfied_list
