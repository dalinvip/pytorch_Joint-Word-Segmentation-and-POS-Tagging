# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:54
# @File : Alphabet.py
# @Last Modify Time : 2018/1/30 15:54
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Alphabet.py
    FUNCTION : None
"""

import os
import sys
import torch
import random
import collections
from DataUtils.Common import seed_num, unkkey, paddingkey, sep, app
torch.manual_seed(seed_num)
random.seed(seed_num)


class CreateAlphabet:
    """
        Class:      Create_Alphabet
        Function:   Build Alphabet By Alphabet Class
        Notice:     The Class Need To Change So That Complete All Kinds Of Tasks
    """
    def __init__(self, min_freq=1, word_min_freq=1, char_min_freq=1, bichar_min_freq=1,
                 train_data=None, dev_data=None, test_data=None, config=None):

        # minimum vocab size
        self.bichar_min_freq = bichar_min_freq
        self.char_min_freq = char_min_freq
        self.word_min_freq = word_min_freq
        self.min_freq = min_freq
        self.config = config
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        # storage word and pos label
        self._word_state = collections.OrderedDict()
        self._char_state = collections.OrderedDict()
        self._bichar_state = collections.OrderedDict()
        self._pos_state = collections.OrderedDict()
        self._word_len_state = collections.OrderedDict()

        # unk and pad
        self._word_state[unkkey] = self.word_min_freq
        self._word_state[paddingkey] = self.word_min_freq
        self._char_state[unkkey] = self.char_min_freq
        self._char_state[paddingkey] = self.char_min_freq
        self._bichar_state[unkkey] = self.bichar_min_freq
        self._bichar_state[paddingkey] = self.bichar_min_freq
        self._pos_state[unkkey] = 1
        self._pos_state[paddingkey] = 1

        # word and label Alphabet
        self.word_alphabet = Alphabet(min_freq=word_min_freq)
        self.char_alphabet = Alphabet(min_freq=char_min_freq)
        self.bichar_alphabet = Alphabet(min_freq=bichar_min_freq)
        # pos min_freq = 1, not cut
        self.pos_alphabet = Alphabet(min_freq=1)
        self.label_alphabet = Alphabet(min_freq=min_freq)
        self.wordLenAlpha = Alphabet()

        # unk key
        self.word_UnkkID = 0
        self.char_UnkID = 0
        self.bichar_UnkID = 0
        self.pos_UnkID = 0

        # padding key
        self.word_PaddingID = 0
        self.char_PaddingID = 0
        self.bichar_PaddingID = 0
        self.pos_PaddingID = 0

        # sep and app
        self.appID = 0
        self.sepID = 0

    @staticmethod
    def _build_data(train_data=None, dev_data=None, test_data=None):
        """
        :param train_data:
        :param dev_data:
        :param test_data:
        :return:
        """
        # handle the data whether to fine_tune
        """
        :param train data:
        :param dev data:
        :param test data:
        :return: merged data
        """
        assert train_data is not None, "The Train Data Is Not Allow Empty."
        datasets = []
        datasets.extend(train_data)
        print("the length of train data {}".format(len(datasets)))
        if dev_data is not None:
            print("the length of dev data {}".format(len(dev_data)))
            datasets.extend(dev_data)
        if test_data is not None:
            print("the length of test data {}".format(len(test_data)))
            datasets.extend(test_data)
        print("the length of data that create Alphabet {}".format(len(datasets)))
        return datasets

    def build_vocab(self):
        """
        :param train_data:
        :param dev_data:
        :param test_data:
        :param debug_index:
        :return:
        """
        train_data = self.train_data
        dev_data = self.dev_data
        test_data = self.test_data
        print("Build Vocab Start...... ")
        datasets = self._build_data(train_data=train_data, dev_data=dev_data, test_data=test_data)

        # create the word Alphabet
        for index, data in enumerate(datasets):
            # word
            for word in data.words:
                if word not in self._word_state:
                    self._word_state[word] = 1
                else:
                    self._word_state[word] += 1
            # char
            for char in data.chars:
                if char not in self._char_state:
                    self._char_state[char] = 1
                else:
                    self._char_state[char] += 1
            # bichar_left
            for bichar in data.bichars_left:
                if bichar not in self._bichar_state:
                    self._bichar_state[bichar] = 1
                else:
                    self._bichar_state[bichar] += 1
            bichar = data.bichars_right[-1]
            if bichar not in self._bichar_state:
                self._bichar_state[bichar] = 1
            else:
                self._bichar_state[bichar] += 1
            # label pos
            for pos in data.pos:
                if pos not in self._pos_state:
                    self._pos_state[pos] = 1
                else:
                    self._pos_state[pos] += 1
            # copy with the gold "SEP#PN"
            for gold in data.gold:
                self.label_alphabet.from_string(gold)

        self.wordLenAlpha.from_string("0")
        self.wordLenAlpha.from_string("1")
        self.wordLenAlpha.from_string("2")
        self.wordLenAlpha.from_string("3")
        self.wordLenAlpha.from_string("4")
        self.wordLenAlpha.from_string("5")
        self.wordLenAlpha.from_string("6")

        # Create id2words and words2id by the Alphabet Class
        self.word_alphabet.initial(self._word_state)
        self.char_alphabet.initial(self._char_state)
        self.bichar_alphabet.initial(self._bichar_state)
        self.pos_alphabet.initial(self._pos_state)

        # unkId and paddingId
        self.word_UnkkID = self.word_alphabet.from_string(unkkey)
        self.char_UnkID = self.char_alphabet.from_string(unkkey)
        self.bichar_UnkID = self.bichar_alphabet.from_string(unkkey)
        self.pos_UnkID = self.pos_alphabet.from_string(unkkey)
        self.word_PaddingID = self.word_alphabet.from_string(paddingkey)
        self.char_PaddingID = self.char_alphabet.from_string(paddingkey)
        self.bichar_PaddingID = self.bichar_alphabet.from_string(paddingkey)
        self.pos_PaddingID = self.pos_alphabet.from_string(paddingkey)

        # copy the app seq ID
        self.appID = self.label_alphabet.from_string(app)
        self.sepID = self.label_alphabet.from_string(sep)

        # fix the vocab
        self.word_alphabet.set_fixed_flag(True)
        self.char_alphabet.set_fixed_flag(True)
        self.bichar_alphabet.set_fixed_flag(True)
        self.pos_alphabet.set_fixed_flag(True)
        self.label_alphabet.set_fixed_flag(True)
        self.wordLenAlpha.set_fixed_flag(True)


class Alphabet:
    """
        Class: Alphabet
        Function: Build vocab
        Params:
              ******    id2words:   type(list),
              ******    word2id:    type(dict)
              ******    vocab_size: vocab size
              ******    min_freq:   vocab minimum freq
              ******    fixed_vocab: fix the vocab after build vocab
              ******    max_cap: max vocab size
    """
    def __init__(self, min_freq=1):
        self.id2words = []
        self.words2id = collections.OrderedDict()
        self.vocab_size = 0
        self.min_freq = min_freq
        self.max_cap = 1e8
        self.fixed_vocab = False

    def initial(self, data):
        """
        :param data:
        :return:
        """
        for key in data:
            if data[key] >= self.min_freq:
                self.from_string(key)
        self.set_fixed_flag(True)

    def set_fixed_flag(self, bfixed):
        """
        :param bfixed:
        :return:
        """
        self.fixed_vocab = bfixed
        if (not self.fixed_vocab) and (self.vocab_size >= self.max_cap):
            self.fixed_vocab = True

    def from_string(self, string):
        """
        :param string:
        :return:
        """
        if string in self.words2id:
            return self.words2id[string]
        else:
            if not self.fixed_vocab:
                newid = self.vocab_size
                self.id2words.append(string)
                self.words2id[string] = newid
                self.vocab_size += 1
                if self.vocab_size >= self.max_cap:
                    self.fixed_vocab = True
                return newid
            else:
                return -1

    def from_id(self, qid, defineStr=""):
        """
        :param qid:
        :param defineStr:
        :return:
        """
        if int(qid) < 0 or self.vocab_size <= qid:
            return defineStr
        else:
            return self.id2words[qid]

    def initial_from_pretrain(self, pretrain_file, unk, padding):
        """
        :param pretrain_file:
        :param unk:
        :param padding:
        :return:
        """
        print("initial alphabet from {}".format(pretrain_file))
        self.loadWord2idAndId2Word(unk)
        self.loadWord2idAndId2Word(padding)
        now_line = 0
        with open(pretrain_file, encoding="UTF-8") as f:
            for line in f.readlines():
                now_line += 1
                sys.stdout.write("\rhandling with {} line".format(now_line))
                info = line.split(" ")
                self.loadWord2idAndId2Word(info[0])
        f.close()
        print("\nHandle Finished.")



