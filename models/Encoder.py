# @Author : bamtercelboo
# @Datetime : 2018/9/4 16:59
# @File : Encoder.py
# @Last Modify Time : 2018/9/4 16:59
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Encoder.py
    FUNCTION : None
"""

import torch.nn
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random
from DataUtils.Common import *
from models.initialize import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Encoder(nn.Module):
    """
        Encoder
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.device = config.device

        # random
        self.char_embed = nn.Embedding(self.config.embed_char_num, self.config.embed_char_dim, sparse=False,
                                       padding_idx=self.config.char_paddingId)
        self.char_embed.weight.requires_grad = True

        self.bichar_embed = nn.Embedding(self.config.embed_bichar_num, self.config.embed_bichar_dim, sparse=False,
                                         padding_idx=self.config.bichar_paddingId)
        self.bichar_embed.weight.requires_grad = True

        # fix the word embedding
        self.static_char_embed = nn.Embedding(self.config.static_embed_char_num, self.config.embed_char_dim, sparse=False,
                                              padding_idx=self.config.static_char_paddingId)
        init_embedding(self.static_char_embed.weight, dim=self.config.embed_char_dim)

        self.static_bichar_embed = nn.Embedding(self.config.static_embed_bichar_num, self.config.embed_bichar_dim, sparse=False,
                                                padding_idx=self.config.static_bichar_paddingId)
        init_embedding(self.static_bichar_embed.weight, dim=self.config.embed_bichar_dim)

        if config.char_pretrained_embed is True:
            self.static_char_embed.weight.data.copy_(self.config.char_pretrain_embed)
            for index in range(self.config.embed_char_dim):
                self.static_char_embed.weight.data[self.config.static_char_paddingId][index] = 0
        self.static_char_embed.weight.requires_grad = False

        if config.bichar_pretrained_embed is True:
            self.static_bichar_embed.weight.data.copy_(self.config.bichar_pretrain_embed)
            for index in range(self.config.embed_bichar_dim):
                self.static_bichar_embed.weight.data[self.config.static_bichar_paddingId][index] = 0
        self.static_bichar_embed.weight.requires_grad = False

        # LSTMCell
        self.lstm_left = nn.LSTMCell(input_size=self.config.rnn_dim, hidden_size=self.config.rnn_hidden_dim, bias=True)
        self.lstm_right = nn.LSTMCell(input_size=self.config.rnn_dim, hidden_size=self.config.rnn_hidden_dim, bias=True)

        # init lstm weight and bias
        init_lstmCell(self.lstm_left, dim=self.config.rnn_hidden_dim)
        init_lstmCell(self.lstm_right, dim=self.config.rnn_hidden_dim)

        self.dropout = nn.Dropout(self.config.dropout)
        self.dropout_embed = nn.Dropout(self.config.dropout_embed)

        self.input_dim = (self.config.embed_char_dim + self.config.embed_bichar_dim) * 2

        self.liner = nn.Linear(in_features=self.input_dim, out_features=self.config.rnn_dim, bias=True)
        init_linear_weight_bias(self.liner)

    def init_cell_hidden(self, batch=1):
        """
        :param batch:  batch size
        :return:
        """
        h = torch.zeros(batch, self.config.rnn_hidden_dim, device=self.device, requires_grad=True)
        c = torch.zeros(batch, self.config.rnn_hidden_dim, device=self.device, requires_grad=True)
        h_c = (h, c)
        return h_c

    def forward(self, features):
        """
        :param features:
        :return:
        """
        batch_length = features.batch_length
        char_features_num = features.static_char_features.size(1)

        char_features = self.char_embed(features.char_features)
        bichar_left_features = self.bichar_embed(features.bichar_left_features)
        bichar_right_features = self.bichar_embed(features.bichar_right_features)

        # fix the word embedding
        static_char_features = self.static_char_embed(features.static_char_features)
        static_bichar_l_features = self.static_bichar_embed(features.static_bichar_left_features)
        static_bichar_r_features = self.static_bichar_embed(features.static_bichar_right_features)

        # dropout
        char_features = self.dropout_embed(char_features)
        bichar_left_features = self.dropout_embed(bichar_left_features)
        bichar_right_features = self.dropout_embed(bichar_right_features)
        static_char_features = self.dropout_embed(static_char_features)
        static_bichar_l_features = self.dropout_embed(static_bichar_l_features)
        static_bichar_r_features = self.dropout_embed(static_bichar_r_features)

        # left concat
        left_concat = torch.cat((char_features, static_char_features, bichar_left_features, static_bichar_l_features), 2)
        # right concat
        right_concat = torch.cat((char_features, static_char_features, bichar_right_features, static_bichar_r_features), 2)

        # non-linear
        left_concat_non_linear = self.dropout(torch.tanh(self.liner(left_concat)))
        left_concat_input = left_concat_non_linear.permute(1, 0, 2)

        right_concat_non_linear = self.dropout(torch.tanh(self.liner(right_concat)))
        right_concat_input = right_concat_non_linear.permute(1, 0, 2)

        left_h, left_c = self.init_cell_hidden(batch_length)
        right_h, right_c = self.init_cell_hidden(batch_length)
        left_lstm_output = []
        right_lstm_output = []
        for idx, id_right in zip(range(char_features_num), reversed(range(char_features_num))):
            left_h, left_c = self.lstm_left(left_concat_input[idx], (left_h, left_c))
            right_h, right_c = self.lstm_right(right_concat_input[id_right], (right_h, right_c))
            left_h = self.dropout(left_h)
            right_h = self.dropout(right_h)
            left_lstm_output.append(left_h.view(batch_length, 1, self.config.rnn_hidden_dim))
            right_lstm_output.insert(0, right_h.view(batch_length, 1, self.config.rnn_hidden_dim))
        left_lstm_output = torch.cat(left_lstm_output, 1)
        right_lstm_output = torch.cat(right_lstm_output, 1)

        encoder_output = torch.cat((left_lstm_output, right_lstm_output), 2)

        return encoder_output


