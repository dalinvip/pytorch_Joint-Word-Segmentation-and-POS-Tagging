# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:55
# @File : Batch_Iterator.py.py
# @Last Modify Time : 2018/1/30 15:55
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Batch_Iterator.py
    FUNCTION : None
"""

import torch
from torch.autograd import Variable
import random

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Batch_Features:
    """
    Batch_Features
    """
    def __init__(self):
        self.batch_length = 0
        self.inst = None
        self.word_features = 0
        self.pos_features = 0
        self.char_features = 0
        self.bichar_left_features = 0
        self.bichar_right_features = 0
        self.static_char_features = 0
        self.static_bichar_left_features = 0
        self.static_bichar_right_features = 0
        self.gold_features = 0

    @staticmethod
    def cuda(features):
        """
        :param features:
        :return:
        """
        features.word_features = features.word_features.cuda()
        features.pos_features = features.pos_features.cuda()
        features.char_features = features.char_features.cuda()
        features.static_char_features = features.static_char_features.cuda()
        features.bichar_left_features = features.bichar_left_features.cuda()
        features.static_bichar_left_features = features.static_bichar_left_features.cuda()
        features.bichar_right_features = features.bichar_right_features.cuda()
        features.static_bichar_right_features = features.static_bichar_right_features.cuda()
        features.gold_features = features.gold_features.cuda()


class Iterators:
    """
    Iterators
    """
    def __init__(self, batch_size=None, data=None, operator=None, operator_static=None,
                 device=None, config=None):
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.data = data
        self.operator = operator
        self.operator_static = operator_static
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []

    def createIterator(self):
        """
        :param batch_size:  batch size
        :param data:  data
        :param operator:
        :param config:
        :return:
        """
        assert isinstance(self.data, list), "ERROR: data must be in list [train_data,dev_data]"
        assert isinstance(self.batch_size, list), "ERROR: batch_size must be in list [16,1,1]"
        for id_data in range(len(self.data)):
            print("*****************    create {} iterator    **************".format(id_data + 1))
            self._convert_word2id(self.data[id_data], self.operator, self.operator_static)
            self.features = self._Create_Each_Iterator(insts=self.data[id_data], batch_size=self.batch_size[id_data],
                                                       operator=self.operator, operator_static=self.operator_static,
                                                       device=self.device)
            self.data_iter.append(self.features)
            self.features = []
        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    @staticmethod
    def _convert_word2id(insts, operator, operator_static):
        """
        :param insts:
        :param operator:
        :param operator_static:
        :return:
        """
        for inst in insts:
            # copy with the word and pos
            for index in range(inst.words_size):
                word = inst.words[index]
                wordID = operator.word_alphabet.from_string(word)
                if wordID == -1:
                    wordID = operator.word_UnkkID
                inst.words_index.append(wordID)

                pos = inst.pos[index]
                posID = operator.pos_alphabet.from_string(pos)
                if posID == -1:
                    posID = operator.pos_UnkID
                inst.pos_index.append(posID)
            # copy with the char
            for index in range(inst.chars_size):
                char = inst.chars[index]
                charID = operator.char_alphabet.from_string(char)
                static_charID = operator_static.char_alphabet.from_string(char)
                if charID == -1:
                    charID = operator.char_UnkID
                if static_charID == -1:
                    static_charID = operator_static.char_UnkID
                inst.chars_index.append(charID)
                inst.static_chars_index.append(static_charID)
            # copy with the bichar_left
            for index in range(inst.bichars_size):
                bichar_left = inst.bichars_left[index]
                bichar_left_ID = operator.bichar_alphabet.from_string(bichar_left)
                static_bichar_left_ID = operator_static.bichar_alphabet.from_string(bichar_left)
                if bichar_left_ID == -1:
                    bichar_left_ID = operator.bichar_UnkID
                if static_bichar_left_ID == -1:
                    static_bichar_left_ID = operator_static.bichar_UnkID
                inst.bichars_left_index.append(bichar_left_ID)
                inst.static_bichars_left_index.append(static_bichar_left_ID)
            # copy with the bichar_right
            for index in range(inst.bichars_size):
                bichar_right = inst.bichars_right[index]
                bichar_right_ID = operator.bichar_alphabet.from_string(bichar_right)
                static_bichar_right_ID = operator_static.bichar_alphabet.from_string(bichar_right)
                if bichar_right_ID == -1:
                    bichar_right_ID = operator.bichar_UnkID
                if static_bichar_right_ID == -1:
                    static_bichar_right_ID = operator_static.bichar_UnkID
                inst.bichars_right_index.append(bichar_right_ID)
                inst.static_bichars_right_index.append(static_bichar_right_ID)
            # copy with the gold
            for index in range(inst.gold_size):
                gold = inst.gold[index]
                goldID = operator.label_alphabet.from_string(gold)
                inst.gold_index.append(goldID)

    def _Create_Each_Iterator(self, insts, batch_size, operator, operator_static, device):
        """
        :param insts:
        :param batch_size:
        :param operator:
        :return:
        """
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1
            # print(batch)
            if len(batch) == batch_size or count_inst == len(insts):
                one_batch = self._Create_Each_Batch(insts=batch, batch_size=batch_size, operator=operator,
                                                    operator_static=operator_static, device=device)
                self.features.append(one_batch)
                batch = []
        print("The all data has created iterator.")
        return self.features

    def _Create_Each_Batch(self, insts, batch_size, operator, operator_static, device):
        """
        :param insts:
        :param batch_size:
        :param operator:
        :param operator_static:
        :return:
        """
        # print("create one batch......")
        batch_length = len(insts)
        # copy with the max length for padding
        max_word_size = -1
        max_char_size = -1
        max_bichar_size = -1
        max_gold_size = -1
        max_pos_size = -1
        for inst in insts:
            word_size = inst.words_size
            if word_size > max_word_size:
                max_word_size = word_size
            char_size = inst.chars_size
            if char_size > max_char_size:
                max_char_size = char_size
            bichar_size = inst.bichars_size
            if bichar_size > max_bichar_size:
                max_bichar_size = bichar_size
            gold_size = inst.gold_size
            if gold_size > max_gold_size:
                max_gold_size = gold_size

        # create with the Tensor/Variable
        # word features
        # batch_word_features = Variable(torch.zeros(batch_length, max_word_size).type(torch.LongTensor))
        # batch_pos_features = Variable(torch.zeros(batch_length, max_word_size).type(torch.LongTensor))
        batch_word_features = torch.zeros(batch_length, max_word_size, device=cpu_device, requires_grad=True).long()
        batch_pos_features = torch.zeros(batch_length, max_word_size, device=cpu_device, requires_grad=True).long()

        # batch_char_features = Variable(torch.zeros(batch_length, max_char_size).type(torch.LongTensor))
        # batch_bichar_left_features = Variable(torch.zeros(batch_length, max_bichar_size).type(torch.LongTensor))
        # batch_bichar_right_features = Variable(torch.zeros(batch_length, max_bichar_size).type(torch.LongTensor))
        batch_char_features = torch.zeros(batch_length, max_char_size, device=cpu_device, requires_grad=True).long()
        batch_bichar_left_features = torch.zeros(batch_length, max_bichar_size, device=cpu_device, requires_grad=True).long()
        batch_bichar_right_features = torch.zeros(batch_length, max_bichar_size, device=cpu_device, requires_grad=True).long()

        # batch_static_char_features = Variable(torch.zeros(batch_length, max_char_size).type(torch.LongTensor))
        # batch_static_bichar_left_features = Variable(torch.zeros(batch_length, max_bichar_size).type(torch.LongTensor))
        # batch_static_bichar_right_features = Variable(torch.zeros(batch_length, max_bichar_size).type(torch.LongTensor))
        batch_static_char_features = torch.zeros(batch_length, max_char_size, device=cpu_device, requires_grad=True).long()
        batch_static_bichar_left_features = torch.zeros(batch_length, max_bichar_size, device=cpu_device, requires_grad=True).long()
        batch_static_bichar_right_features = torch.zeros(batch_length, max_bichar_size, device=cpu_device, requires_grad=True).long()

        # batch_gold_features = Variable(torch.zeros(max_gold_size * batch_length).type(torch.LongTensor))
        batch_gold_features = torch.zeros(max_gold_size * batch_length, device=cpu_device, requires_grad=True).long()

        for id_inst in range(batch_length):
            inst = insts[id_inst]
            # copy with the word features
            for id_word_index in range(max_word_size):
                if id_word_index < inst.words_size:
                    batch_word_features.data[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    batch_word_features.data[id_inst][id_word_index] = operator.word_PaddingID

            # copy with the pos features
            for id_pos_index in range(max_word_size):
                if id_pos_index < inst.words_size:
                    batch_pos_features.data[id_inst][id_pos_index] = inst.pos_index[id_pos_index]
                else:
                    batch_pos_features.data[id_inst][id_pos_index] = operator.pos_PaddingID

            # copy with the char features
            for id_char_index in range(max_char_size):
                if id_char_index < inst.chars_size:
                    batch_char_features.data[id_inst][id_char_index] = inst.chars_index[id_char_index]
                    batch_static_char_features.data[id_inst][id_char_index] = inst.static_chars_index[id_char_index]
                else:
                    batch_char_features.data[id_inst][id_char_index] = operator.char_PaddingID
                    batch_static_char_features.data[id_inst][id_char_index] = operator_static.char_PaddingID

            # copy with the bichar_left features
            for id_bichar_left_index in range(max_bichar_size):
                if id_bichar_left_index < inst.bichars_size:
                    batch_bichar_left_features.data[id_inst][id_bichar_left_index] = inst.bichars_left_index[id_bichar_left_index]
                    batch_static_bichar_left_features.data[id_inst][id_bichar_left_index] = int(inst.static_bichars_left_index[id_bichar_left_index])
                else:
                    batch_bichar_left_features.data[id_inst][id_bichar_left_index] = operator.bichar_PaddingID
                    batch_static_bichar_left_features.data[id_inst][id_bichar_left_index] = int(operator_static.bichar_PaddingID)

            # copy with the bichar_right features
            for id_bichar_right_index in range(max_bichar_size):
                if id_bichar_right_index < inst.bichars_size:
                    batch_bichar_right_features.data[id_inst][id_bichar_right_index] = inst.bichars_right_index[id_bichar_right_index]
                    batch_static_bichar_right_features.data[id_inst][id_bichar_right_index] = inst.static_bichars_right_index[id_bichar_right_index]
                else:
                    batch_bichar_right_features.data[id_inst][id_bichar_right_index] = operator.bichar_PaddingID
                    batch_static_bichar_right_features.data[id_inst][id_bichar_right_index] = operator_static.bichar_PaddingID

            # copy with the gold features
            for id_gold_index in range(max_gold_size):
                if id_gold_index < inst.gold_size:
                    batch_gold_features.data[id_gold_index + id_inst * max_gold_size] = inst.gold_index[id_gold_index]
                else:
                    batch_gold_features.data[id_gold_index + id_inst * max_gold_size] = 0

        # batch
        features = Batch_Features()
        features.batch_length = batch_length
        features.inst = insts
        features.word_features = batch_word_features
        features.pos_features = batch_pos_features
        features.char_features = batch_char_features
        features.static_char_features = batch_static_char_features
        features.bichar_left_features = batch_bichar_left_features
        features.static_bichar_left_features = batch_static_bichar_left_features
        features.bichar_right_features = batch_bichar_right_features
        features.static_bichar_right_features = batch_static_bichar_right_features
        features.gold_features = batch_gold_features

        if self.config.device != cpu_device:
            features.cuda(features)
        return features

