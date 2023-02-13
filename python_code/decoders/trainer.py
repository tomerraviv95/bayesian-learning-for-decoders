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
        self.channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                                   pilots_length=conf.pilot_size,
                                                   blocks_num=conf.blocks_num)

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

    def evaluate(self) -> Tuple[List[float], List[float], List[float]]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: list of ber per timestep
        """
        total_ber = []
        # draw words for a given snr
        transmitted_words, received_words = self.channel_dataset.__getitem__(snr_list=[conf.snr])
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            # get current word and channel
            tx, rx = transmitted_words[block_ind], received_words[block_ind]
            # split words into data and pilot part
            tx_pilot, tx_data = tx[:conf.pilot_size], tx[conf.pilot_size:]
            rx_pilot, rx_data = rx[:conf.pilot_size], rx[conf.pilot_size:]
            if conf.is_online_training:
                # re-train the detector
                self._online_training(tx_pilot, rx_pilot)
            # detect data part after training on the pilot part
            decoded_words = self.forward(rx_data)
            # calculate accuracy
            ber = calculate_ber(decoded_words, tx_data)
            print(f'current: {block_ind, ber}')
            total_ber.append(ber)

        print(f'Final ser: {sum(total_ber) / len(total_ber)}')
        return total_ber, correct_values_list, error_values_list

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self.calc_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss
