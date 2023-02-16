import torch
from torch import nn
from torch.nn.parameter import Parameter

from python_code import DEVICE
from python_code.decoders.bp_nn_weights import init_w_skipconn2even, \
    initialize_w_v2c
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy


class BayesianOddLayer(torch.nn.Module):
    def __init__(self, clip_tanh, input_output_layer_size, neurons, code_pcm):
        super(BayesianOddLayer, self).__init__()
        w_skipconn2even, w_skipconn2even_mask = init_w_skipconn2even(input_output_layer_size=input_output_layer_size,
                                                                     neurons=neurons,
                                                                     code_pcm=code_pcm)
        w_odd2even, w_odd2even_mask = initialize_w_v2c(neurons=neurons, code_pcm=code_pcm)
        self.odd_weights = Parameter(w_odd2even.to(DEVICE))
        self.llr_weights = Parameter(w_skipconn2even.to(DEVICE))
        self.w_odd2even_mask = w_odd2even_mask.to(device=DEVICE)
        self.dropout_logit = nn.Parameter(torch.rand(w_odd2even_mask.shape[0])).to(DEVICE)
        self.w_skipconn2even_mask = w_skipconn2even_mask.to(device=DEVICE)
        self.clip_tanh = clip_tanh
        self.kl_scale = 5

    def forward(self, x, llr, llr_mask_only=False):
        self.u = torch.rand(self.odd_weights.shape).to(DEVICE)
        total_mask = self.w_odd2even_mask * self.odd_weights
        mask_after_dropout = dropout_ori(total_mask, self.dropout_logit, self.u)
        odd_weights_times_messages_after_dropout = torch.matmul(x, mask_after_dropout)
        mask_after_dropout_tilde = dropout_tilde(total_mask, self.dropout_logit, self.u)
        odd_weights_times_messages_tilde = torch.matmul(x, mask_after_dropout_tilde)
        if llr_mask_only:
            odd_weights_times_llr = torch.matmul(llr, self.w_skipconn2even_mask)
        else:
            odd_weights_times_llr = torch.matmul(llr, self.w_skipconn2even_mask * self.llr_weights)
        odd_clamp = torch.clamp(odd_weights_times_messages_after_dropout + odd_weights_times_llr,
                                min=-self.clip_tanh, max=self.clip_tanh)
        output = torch.tanh(0.5 * odd_clamp)
        # computation for ARM loss
        odd_clamp_tilde = torch.clamp(odd_weights_times_messages_tilde + odd_weights_times_llr,
                                      min=-self.clip_tanh, max=self.clip_tanh)
        self.init_arm_tilde = torch.tanh(0.5 * odd_clamp_tilde)
        scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logit).reshape(-1))
        first_layer_kl = scaling1 * torch.norm(self.odd_weights, dim=1) ** 2
        H1 = entropy(torch.sigmoid(self.dropout_logit).reshape(-1))
        self.kl_term = torch.mean(first_layer_kl - H1)
        return output
