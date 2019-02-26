# @Author : bamtercelboo
# @Datetime : 2018/9/4 17:00
# @File : Decoder.py
# @Last Modify Time : 2018/9/4 17:00
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Decoder.py
    FUNCTION : None
"""

import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np
import random
from DataUtils.state import state_batch_instance
from DataUtils.Common import *
from models.initialize import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Decoder(nn.Module):
    """
        Decoder
    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.device = config.device

        self.pos_paddingKey = self.config.create_alphabet.pos_PaddingID

        self.lstmcell = nn.LSTMCell(input_size=self.config.rnn_dim, hidden_size=self.config.rnn_dim, bias=True)
        init_lstmCell(self.lstmcell, dim=self.config.rnn_dim)
        # init.xavier_uniform_(self.lstmcell.weight_ih)
        # init.xavier_uniform_(self.lstmcell.weight_hh)
        # self.lstmcell.bias_hh.data.uniform_(-np.sqrt(6 / (self.config.rnn_dim + 1)), np.sqrt(6 / (self.config.rnn_dim + 1)))
        # self.lstmcell.bias_ih.data.uniform_(-np.sqrt(6 / (self.config.rnn_dim + 1)), np.sqrt(6 / (self.config.rnn_dim + 1)))

        self.pos_embed = nn.Embedding(num_embeddings=self.config.pos_size, embedding_dim=self.config.pos_dim,
                                      padding_idx=self.pos_paddingKey)
        init_embedding(self.pos_embed.weight, dim=self.config.pos_dim, paddingKey=self.pos_paddingKey)
        # init.uniform_(self.pos_embed.weight, a=-np.sqrt(3 / self.config.pos_dim), b=np.sqrt(3 / self.config.pos_dim))
        # for i in range(self.config.pos_dim):
        #     self.pos_embed.weight.data[self.pos_paddingKey][i] = 0
        # self.pos_embed.weight.requires_grad = True

        self.linear = nn.Linear(in_features=self.config.rnn_hidden_dim * 2 + self.config.rnn_dim,
                                out_features=self.config.label_size, bias=False)
        # init_linear_weight_bias(self.linear)

        self.combine_linear = nn.Linear(in_features=self.config.rnn_hidden_dim * 2 + self.config.pos_dim,
                                        out_features=self.config.rnn_dim, bias=True)

        # init.xavier_uniform_(self.linear.weight)
        init_linear_weight_bias(self.linear)
        init_linear_weight_bias(self.combine_linear)
        # init.xavier_uniform_(self.combine_linear.weight)
        # self.combine_linear.bias.data.uniform_(-np.sqrt(6 / (self.config.rnn_dim + 1)), np.sqrt(6 / (self.config.rnn_dim + 1)))

        self.dropout = nn.Dropout(self.config.dropout)

        self.softmax = nn.LogSoftmax(dim=1)

        self.bucket = torch.zeros(1, self.config.label_size, device=self.device, requires_grad=True)
        self.bucket_rnn = torch.zeros(1, self.config.rnn_hidden_dim, device=self.device, requires_grad=True)

    def init_hidden_cell(self, batch_size):
        """
        :param batch_size:  batch size
        :return:
        """
        z_bucket = torch.zeros(batch_size, self.config.rnn_dim, device=self.device, requires_grad=True)
        h_bucket = torch.zeros(batch_size, self.config.rnn_hidden_dim, device=self.device, requires_grad=True)
        c_bucket = torch.zeros(batch_size, self.config.rnn_hidden_dim, device=self.device, requires_grad=True)
        # if self.device != cpu_device:
        #     h_bucket, c_bucket, z_bucket = h_bucket.cuda(), c_bucket.cuda(), z_bucket.cuda()
        return h_bucket, c_bucket, z_bucket

    def forward(self, features, encoder_out, train=False):
        """
        :param features:
        :param encoder_out:  Encoder output
        :param train:  train for dropout
        :return:
        """

        batch_length = features.batch_length
        encoder_out = encoder_out.permute(1, 0, 2)
        char_features_num = encoder_out.size(0)
        state = state_batch_instance(features, char_features_num)
        # state.show()
        char_output = []
        for id_char in range(char_features_num):
            h_now, c_now = self.batch_wordLstm(id_char, batch_length, encoder_out, state)

            v = torch.cat((h_now, encoder_out[id_char]), 1)
            # print(v)
            output = self.linear(v)
            if id_char is 0:
                for i in range(batch_length):
                    output.data[i][self.config.create_alphabet.appID] = -10e9
            self.batch_action(state, id_char, output, h_now, c_now, batch_length, train)
            char_output.append(output.unsqueeze(1))
        decoder_out = torch.cat(char_output, 1)
        decoder_out = decoder_out.view(batch_length * char_features_num, -1)
        decoder_out = self.softmax(decoder_out)
        return decoder_out, state

    def batch_wordLstm(self, id_char, batch_length, encoder_out, state):
        """
        :param id_char:  id word
        :param batch_length:  batch count
        :param encoder_out:  Encoder output
        :param state:  Decoder state
        :return:
        """
        if id_char is 0:
            h, c, z = self.init_hidden_cell(batch_length)
        else:
            h, c = state.word_hiddens[-1], state.word_cells[-1]
            # copy with the pos features
            # last_pos = torch.zeros(batch_length, device=self.device, requires_grad=True).long()
            last_pos = torch.zeros(batch_length, device=self.device, requires_grad=True).long()
            # if self.device != cpu_device:
            #     last_pos = last_pos.cuda()
            pos_id_array = np.array(state.pos_id[-1])
            last_pos.data.copy_(torch.from_numpy(pos_id_array))
            last_pos_embed = self.dropout(self.pos_embed(last_pos))

            # copy with the word features
            batch_char_embed = []
            for id_batch, id_batch_value in enumerate(state.words_startindex[-1]):
                chars_embed = []
                last_word_len = 0
                if id_batch_value is -1:
                    # word_bucket = torch.zeros(1, 2 * self.config.rnn_hidden_dim, device=self.device, requires_grad=True)
                    word_bucket = torch.zeros(1, 2 * self.config.rnn_hidden_dim, device=self.device, requires_grad=True)
                    # if self.device != cpu_device:
                    #     word_bucket = word_bucket.cuda()
                    batch_char_embed.append(word_bucket)
                    continue
                last_word_len = id_char - id_batch_value
                chars_embed.append((encoder_out.permute(1, 0, 2)[id_batch][id_batch_value:id_char].unsqueeze(0)))
                # chars_embed = torch.cat(list(encoder_out.permute(1, 0, 2)[id_batch][id_batch_value:id_char].view(1, last_word_len, 2 * self.config.rnn_hidden_dim)), 1)
                chars_embed = torch.cat(chars_embed, 1).permute(0, 2, 1)
                last_word_embed = F.avg_pool1d(chars_embed, chars_embed.size(2)).squeeze(2)
                batch_char_embed.append(last_word_embed)
            batch_char_embed = torch.cat(batch_char_embed, 0)
            concat = torch.cat((last_pos_embed, batch_char_embed), 1)
            z = self.dropout(torch.tanh(self.combine_linear(concat)))
        h_now, c_now = self.lstmcell(z, (h, c))

        return h_now, c_now

    def batch_action(self, state, index, output, hidden_now, cell_now, batch_length, train):
        """
        :param state:  decoder state
        :param index:  index
        :param output:  output
        :param hidden_now:  lstm hidden
        :param cell_now:  lstm cell
        :param batch_length:  batch count
        :param train:  train for decoder
        :return:
        """
        action = []
        if train:
            for i in range(batch_length):
                if index < len(state.gold[i]):
                    action.append(state.gold[i][index])
                else:
                    action.append("ACT")
        else:
            actionID_list = self.torch_max(output)
            for actionID in actionID_list:
                action.append(self.config.create_alphabet.label_alphabet.from_id(actionID))
        state.actions.append(action)

        pos_id = []
        start_index = []
        for id_batch, act in enumerate(action):
            pos = act.find("#")
            if pos == -1:
                # app
                if index < len(state.chars[id_batch]):
                    state.words[id_batch][-1] += (state.chars[id_batch][index])
                    start_index.append((index + 1) - len(state.words[id_batch][-1]))
                else:
                    start_index.append(-1)

                if act == app:
                    pos_id.append(state.pos_id[-1][id_batch])
                elif act == "ACT":
                    pos_id.append(self.pos_paddingKey)
            else:
                posLabel = act[pos + 1:]
                if index < len(state.chars[id_batch]):
                    temp_word = state.chars[id_batch][index]
                    state.words[id_batch].append(temp_word)
                    state.pos_labels[id_batch].append(posLabel)
                    start_index.append((index + 1) - len(state.words[id_batch][-1]))
                else:
                    start_index.append(-1)
                posId = self.config.create_alphabet.pos_alphabet.loadWord2idAndId2Word(posLabel)
                pos_id.append(posId)

        state.words_startindex.append(start_index)
        state.pos_id.append(pos_id)
        state.word_cells.append(cell_now)
        state.word_hiddens.append(hidden_now)

    @staticmethod
    def getMaxindex(decoder_output):
        """
        :param decoder_output:
        :return:
        """
        decoder_output_list = decoder_output.data.tolist()
        maxIndex = decoder_output_list.index(np.max(decoder_output_list))
        return maxIndex

    @staticmethod
    def torch_max(output):
        """
        :param output: batch * seq_len * label_num
        :return:
        """
        _, arg_max = torch.max(output, dim=1)
        return arg_max.data.cpu().numpy().tolist()



