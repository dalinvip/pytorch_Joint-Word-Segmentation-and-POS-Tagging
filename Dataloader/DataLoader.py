# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import re
import random
import torch
import unicodedata
from Dataloader.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    def _sort(self, insts, path):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        # self._sort2file(sorted_insts, path)
        return sorted_insts

    def _sort2file(self, insts, path):
        print("Sort Result To File.")
        # path = path + "_sort.txt"
        print(path)
        exit()


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """
    def __init__(self, path, shuffle, config):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        #
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle

    def dataLoader(self):
        """
        :return:
        """
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self._Load_Each_Data(path=path[id_data], shuffle=shuffle)
            if shuffle is True and id_data == 0:
                print("shuffle train data......")
                random.shuffle(insts)
            # insts = self._sort(insts, path=path[id_data])

            self.data_list.append(insts)
        # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_Data(self, path=None, shuffle=False):
        """
        :param path:
        :param shuffle:
        :return:
        """
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        with open(path, encoding="UTF-8") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                # copy with "/n"
                line = unicodedata.normalize('NFKC', line.strip())
                # init instance
                inst = Instance()
                line = line.split(" ")
                inst.line = " ".join(line)
                # print(inst.line)
                # print(line)
                count = 0
                for word_pos in line:
                    # segment the word and pos in line
                    word, _, label = word_pos.partition("_")
                    word_length = len(word)
                    inst.words.append(word)
                    inst.gold_seg.append("[" + str(count) + "," + str(count + word_length) + "]")
                    inst.gold_pos.append("[" + str(count) + "," + str(count + word_length) + "]" + label)
                    count += word_length
                    for i in range(word_length):
                        char = word[i]
                        # print(char)
                        inst.chars.append(char)
                        if i == 0:
                            inst.gold.append(sep + "#" + label)
                            inst.pos.append(label)
                        else:
                            inst.gold.append(app)
                char_number = len(inst.chars)
                for i in range(char_number):
                    # copy with the left bichars
                    if i is 0:
                        inst.bichars_left.append(nullkey + inst.chars[i])
                    else:
                        inst.bichars_left.append(inst.chars[i - 1] + inst.chars[i])
                    # copy with the right bichars
                    if i == char_number - 1:
                        inst.bichars_right.append(inst.chars[i] + nullkey)
                    else:
                        inst.bichars_right.append(inst.chars[i] + inst.chars[i + 1])
                # char/word size
                inst.chars_size = len(inst.chars)
                inst.words_size = len(inst.words)
                inst.bichars_size = len(inst.bichars_left)
                inst.gold_size = len(inst.gold)
                # add one inst that represent one sentence into the list
                insts.append(inst)
                if len(insts) == self.max_count:
                    break
        return insts

