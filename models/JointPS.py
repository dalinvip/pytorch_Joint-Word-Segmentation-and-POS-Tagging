# @Author : bamtercelboo
# @Datetime : 2018/9/4 16:48
# @File : JointPS.py
# @Last Modify Time : 2018/9/4 16:48
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  JointPS.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import random
import numpy as np
import time
from models.Encoder import Encoder
from models.Decoder import Decoder
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class JointPS(nn.Module):
    """
        JointPS
    """

    def __init__(self, config):
        super(JointPS, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, train=False):
        """
        :param x:
        :param train:
        :return:
        """
        encoder = self.encoder(x)
        decoder_out, state = self.decoder(x, encoder, train=train)
        return decoder_out, state


