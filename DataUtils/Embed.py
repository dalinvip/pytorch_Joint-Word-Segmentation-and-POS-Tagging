# @Author : bamtercelboo
# @Datetime : 2018/8/27 15:34
# @File : Embed.py
# @Last Modify Time : 2018/8/27 15:34
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Embed.py
    FUNCTION : None
"""

import os
import sys
import time
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from DataUtils.Common import *
torch.manual_seed(seed_num)
np.random.seed(seed_num)


class Embed(object):
    """
    Embed
    """
    def __init__(self, path, words_dict, embed_type, pad):
        self.embed_type_enum = ["zero", "avg", "uniform", "nn"]
        self.path = path
        self.words_dict = words_dict
        self.embed_type = embed_type
        self.pad = pad
        # print(self.words_dict)
        if not isinstance(self.words_dict, dict):
            self.words_dict, self.words_list = self._list2dict(self.words_dict)
        if pad is not None: self.padID = self.words_dict[pad]
        # print(self.words_dict)
        self.dim, self.words_count = self._get_dim(path=self.path), len(self.words_dict)
        self.exact_count, self.fuzzy_count, self.oov_count = 0, 0, 0

    def get_embed(self):
        """
        :return:
        """
        embed_dict = None
        if self.embed_type in self.embed_type_enum:
            embed_dict = self._read_file(path=self.path)
        else:
            print("embed_type illegal, must be in {}".format(self.embed_type_enum))
            exit()
        # print(embed_dict)
        embed = None
        if self.embed_type == "nn":
            embed = self._nn_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == "zero":
            embed = self._zeros_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == "uniform":
            embed = self._uniform_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == "avg":
            embed = self._avg_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        # print(embed)
        self.info()
        return embed

    def _zeros_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        """
        print("loading pre_train embedding by zeros for out of vocabulary.")
        embeddings = np.zeros((int(self.words_count), int(self.dim)))
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _nn_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        """
        print("loading pre_train embedding by nn.Embedding for out of vocabulary.")
        embed = nn.Embedding(int(self.words_count), int(self.dim))
        init.xavier_uniform(embed.weight.data)
        embeddings = np.array(embed.weight.data)
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _uniform_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        """
        print("loading pre_train embedding by uniform for out of vocabulary.")
        embeddings = np.zeros((int(self.words_count), int(self.dim)))
        inword_list = {}
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        uniform_col = np.random.uniform(-0.25, 0.25, int(self.dim)).round(6)  # uniform
        for i in range(len(words_dict)):
            if i not in inword_list and i != self.padID:
                embeddings[i] = uniform_col
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _avg_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        """
        print("loading pre_train embedding by avg for out of vocabulary.")
        embeddings = np.zeros((int(self.words_count), int(self.dim)))
        inword_list = {}
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        sum_col = np.sum(embeddings, axis=0) / len(inword_list)  # avg
        for i in range(len(words_dict)):
            if i not in inword_list and i != self.padID:
                embeddings[i] = sum_col
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    @staticmethod
    def _read_file(path):
        """
        :param path: embed file path
        :return:
        """
        embed_dict = {}
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            lines = tqdm.tqdm(lines)
            for line in lines:
                values = line.strip().split(' ')
                if len(values) == 1 or len(values) == 2 or len(values) == 3:
                    continue
                w, v = values[0], values[1:]
                embed_dict[w] = v
        return embed_dict

    def info(self):
        """
        :return:
        """
        total_count = self.exact_count + self.fuzzy_count
        print("Words count {}, Embed dim {}.".format(self.words_count, self.dim))
        print("Exact count {} / {}".format(self.exact_count, self.words_count))
        print("Fuzzy count {} / {}".format(self.fuzzy_count, self.words_count))
        print("  INV count {} / {}".format(total_count, self.words_count))
        print("  OOV count {} / {}".format(self.oov_count, self.words_count))
        print("  OOV radio ===> {}%".format(np.round((self.oov_count / total_count) * 100, 2)))
        print(40 * "*")

    @staticmethod
    def _get_dim(path):
        """
        :param path:
        :return:
        """
        embedding_dim = -1
        with open(path, encoding='utf-8') as f:
            for line in f:
                line_split = line.strip().split(' ')
                if len(line_split) == 1:
                    embedding_dim = line_split[0]
                    break
                elif len(line_split) == 2:
                    embedding_dim = line_split[1]
                    break
                else:
                    embedding_dim = len(line_split) - 1
                    break
        return embedding_dim

    @staticmethod
    def _list2dict(convert_list):
        """
        :param convert_list:
        :return:
        """
        list_dict = OrderedDict()
        list_lower = []
        for index, word in enumerate(convert_list):
            list_lower.append(word.lower())
            list_dict[word] = index
        assert len(list_lower) == len(list_dict)
        return list_dict, list_lower

