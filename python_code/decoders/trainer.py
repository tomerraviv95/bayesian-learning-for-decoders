import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.optim import RMSprop, Adam, SGD

from dir_definitions import ECC_MATRICES_DIR
from python_code import DEVICE, conf
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.constants import TANNER_GRAPH_CYCLE_REDUCTION
from python_code.utils.metrics import calculate_ber
from python_code.utils.python_utils import load_code_parameters

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Trainer(nn.Module):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self):
        super(Trainer, self).__init__()
        # initialize matrices, datasets and detector
        self._initialize_dataloader()
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference
        self.odd_llr_mask_only = True
        self.even_mask_only = True
        self.multiloss_output_mask_only = True
        self.filter_in_iterations_eval = True
        self.output_mask_only = False
        self.multi_loss_flag = True
        self.iteration_num = conf.iterations
        self._code_bits = conf.code_bits
        self._info_bits = conf.info_bits
        self.code_pcm, self.code_gm = load_code_parameters(self._code_bits, self._info_bits,
                                                           ECC_MATRICES_DIR, TANNER_GRAPH_CYCLE_REDUCTION)
        self.neurons = int(np.sum(self.code_pcm))

    def get_name(self):
        return self.__name__()

    # calculate train loss
    def calc_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self, lr: float):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                 lr=lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        elif conf.loss_type == 'BCEWithLogits':
            self.criterion = BCEWithLogitsLoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.train_channel_dataset = ChannelModelDataset(block_size=conf.train_block_size)
        self.val_channel_dataset = ChannelModelDataset(block_size=conf.val_block_size)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def forward(self, rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every trainer must have some forward pass for its detector
        """
        pass

    def train_and_eval(self):
        for block_ind in range(conf.train_blocks_num):
            print(f'Training on block {block_ind}')
            # draw words from channel
            tx, rx = self.channel_dataset.__getitem__(
                snr_list=list(range(conf.train_snr_start, conf.train_snr_end + 1)))
            # train the decoder
            self._online_training(tx, rx)
            print('Evaluating...')
            self.eval()

    def eval(self) -> List[float]:
        """
        The evaluation running on multiple pairs of transmitted and received blocks.
        :return: list of ber per block
        """
        total_ber = []
        for block_ind in range(conf.val_blocks_num):
            print('*' * 20)
            tx, rx = self.channel_dataset.__getitem__(snr_list=[conf.val_snr])
            # detect data part after training on the pilot part
            decoded_words = self.forward(rx)
            # calculate accuracy
            ber = calculate_ber(decoded_words, tx)
            print(f'current: {block_ind, ber}')
            total_ber.append(ber)
        print(f'Final ser: {sum(total_ber) / len(total_ber)}')
        return total_ber

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self.calc_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss
