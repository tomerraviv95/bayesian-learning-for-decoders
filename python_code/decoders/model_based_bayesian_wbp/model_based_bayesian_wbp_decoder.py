import torch

from python_code.decoders.bp_nn import InputLayer, EvenLayer, OutputLayer
from python_code.decoders.model_based_bayesian_wbp.bayesian_bp_nn import BayesianOddLayer
from python_code.decoders.trainer import Trainer
from python_code.utils.constants import HALF, CLIPPING_VAL, Phase

EPOCHS = 100
BATCH_SIZE = 120


class ModelBasedBayesianWBPDecoder(Trainer):
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
        return 'Model-Based Bayesian WBP Decoder'

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
        # initialize
        even_output = self.input_layer.forward(rx)
        for i in range(0, self.iteration_num - 1):
            for e in range(EPOCHS):
                # select BATCH_SIZE samples randomly
                idx = torch.randperm(tx.shape[0])[:BATCH_SIZE]
                cur_tx, cur_even_output, cur_rx = tx[idx], even_output[idx], rx[idx]
                arm_original, arm_tilde, u_list, kl_term = [], [], [], 0
                total_even_output = 0
                for _ in range(self.ensemble_num):
                    # odd - variables to check
                    odd_output = self.odd_layer.forward(cur_even_output, cur_rx, llr_mask_only=self.odd_llr_mask_only)
                    # even - check to variables
                    cur_even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                    cur_output = cur_rx + self.output_layer.forward(cur_even_output, mask_only=self.output_mask_only)
                    arm_original.append(cur_output)
                    even_output_tilde = self.odd_layer.init_arm_tilde
                    output_tilde = cur_rx + self.output_layer.forward(even_output_tilde,
                                                                      mask_only=self.output_mask_only)
                    arm_tilde.append(output_tilde)
                    u_list.append(self.odd_layer.u)
                    total_even_output += cur_even_output
                avg_even_output = total_even_output / self.ensemble_num
                output = cur_rx + self.output_layer.forward(avg_even_output, mask_only=self.output_mask_only)
                kl_term = self.odd_layer.kl_term
                self.calc_loss(cur_tx, output, arm_original, arm_tilde, u_list, self.odd_layer.dropout_logit, kl_term)
            with torch.no_grad():
                total_even_output = 0
                for _ in range(self.ensemble_num):
                    # odd - variables to check
                    odd_output = self.odd_layer.forward(even_output, rx, llr_mask_only=self.odd_llr_mask_only)
                    # even - check to variables
                    cur_even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                    total_even_output += cur_even_output
                even_output = total_even_output / self.ensemble_num

    def forward(self, x, phase: Phase):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        output_list = [0] * (self.iteration_num)

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = x + self.output_layer.forward(even_output, mask_only=self.output_layer)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(0, self.iteration_num - 1):
            total_even_output = 0
            for _ in range(self.ensemble_num):
                # odd - variables to check
                odd_output = self.odd_layer.forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
                # even - check to variables
                even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                total_even_output += even_output
            avg_even_output = total_even_output / self.ensemble_num
            # output layer
            output = x + self.output_layer.forward(avg_even_output, mask_only=self.output_mask_only)
            output_list[i + 1] = output.clone()

        return output_list
