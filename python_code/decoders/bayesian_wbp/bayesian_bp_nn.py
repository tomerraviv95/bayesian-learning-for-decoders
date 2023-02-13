import torch
from torch import nn
from torch.nn.parameter import Parameter

from python_code import DEVICE
from python_code.decoders.bp_nn_weights import initialize_w_init, initialize_w_c2v, init_w_skipconn2even, \
    initialize_w_v2c, init_w_output
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InputLayer(torch.nn.Module):
    def __init__(self, input_output_layer_size, neurons, code_pcm, clip_tanh, bits_num):
        super(InputLayer, self).__init__()
        self.input_weights = initialize_w_init(input_output_layer_size=input_output_layer_size, neurons=neurons,
                                               code_pcm=code_pcm).to(device=device)
        self.neurons = neurons
        self.clip_tanh = clip_tanh
        self.N = bits_num

    def forward(self, x):
        repeated_x = x.repeat(1, self.neurons)
        unsqueeze_flattened_input_weights = torch.reshape(torch.t(self.input_weights), (-1,))
        x_times_input_weight = torch.mul(repeated_x, unsqueeze_flattened_input_weights).reshape(x.shape[0],
                                                                                                self.neurons,
                                                                                                self.N)
        u = torch.tanh(0.5 * torch.clamp(x_times_input_weight, min=-self.clip_tanh, max=self.clip_tanh))
        z_input = torch.prod(u + torch.exp(-1000000 * torch.abs(u)), dim=2)
        return torch.log(torch.div(1 + z_input + 1e-8, 1 - z_input + 1e-8))


class EvenLayer(torch.nn.Module):
    def __init__(self, clip_tanh, neurons, code_pcm):
        super(EvenLayer, self).__init__()
        w_even2odd, w_even2odd_mask = initialize_w_c2v(neurons=neurons, code_pcm=code_pcm)
        if torch.cuda.is_available():
            self.even_weights = Parameter(w_even2odd.cuda())
        else:
            self.even_weights = Parameter(w_even2odd)
        self.w_even2odd_mask = w_even2odd_mask.to(device=device)
        self.neurons = neurons
        self.clip_tanh = clip_tanh
        self.dropout_logit = nn.Parameter(torch.rand(self.neurons).reshape(1, -1))

    def forward(self, x, mask_only=False):
        log_abs_x = torch.log(torch.abs(x) + 1e-8)
        x_b = 0.5 - 0.5 * torch.sign(x)
        syndrome = torch.matmul(x_b, self.w_even2odd_mask) % 2
        bipolar_syndrome = 1 - 2 * syndrome
        if mask_only:
            log_abs_x_tanner = torch.matmul(log_abs_x, self.w_even2odd_mask)
        else:
            log_abs_x_tanner = torch.matmul(log_abs_x, self.w_even2odd_mask * self.even_weights)
        even_prod_messages = bipolar_syndrome * torch.exp(log_abs_x_tanner)
        return torch.log(torch.div((1 + even_prod_messages + 1e-8), (1 - even_prod_messages + 1e-8)))


class OddLayer(torch.nn.Module):
    def __init__(self, clip_tanh, input_output_layer_size, neurons, code_pcm):
        super(OddLayer, self).__init__()
        w_skipconn2even, w_skipconn2even_mask = init_w_skipconn2even(input_output_layer_size=input_output_layer_size,
                                                                     neurons=neurons,
                                                                     code_pcm=code_pcm)
        w_odd2even, w_odd2even_mask = initialize_w_v2c(neurons=neurons, code_pcm=code_pcm)
        if torch.cuda.is_available():
            self.odd_weights = Parameter(w_odd2even.cuda())
            self.llr_weights = Parameter(w_skipconn2even.cuda())
        else:
            self.odd_weights = Parameter(w_odd2even)
            self.llr_weights = Parameter(w_skipconn2even)
        self.w_odd2even_mask = w_odd2even_mask.to(device=device)
        self.dropout_logit = nn.Parameter(torch.rand(w_odd2even_mask.shape[0])).to(DEVICE)
        self.w_skipconn2even_mask = w_skipconn2even_mask.to(device=device)
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


class OutputLayer(torch.nn.Module):
    def __init__(self, neurons, input_output_layer_size, code_pcm):
        super(OutputLayer, self).__init__()
        w_output, w_output_mask = init_w_output(neurons=neurons, input_output_layer_size=input_output_layer_size,
                                                code_pcm=code_pcm)
        if torch.cuda.is_available():
            self.output_weights = Parameter(w_output.cuda())
        else:
            self.output_weights = Parameter(w_output)
        self.w_output_mask = w_output_mask.to(device=device)

    def forward(self, x, mask_only=False):
        if mask_only:
            return torch.matmul(x, self.w_output_mask)
        return torch.matmul(x, self.w_output_mask * self.output_weights)
