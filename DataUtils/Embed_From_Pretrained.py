# @Author : bamtercelboo
# @Datetime : 2018/2/3 14:03
# @File : Embed_From_Pretrained.py
# @Last Modify Time : 2018/2/3 14:03
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Embed_From_Pretrained.py
    FUNCTION : None
"""

import os
import sys
import torch
import torch.nn.init as init
import numpy as np
import random
import torch.nn as nn
import hyperparams as hy
torch.manual_seed(hy.seed_num)
random.seed(hy.seed_num)


def Pretrain_Embed(file, vocab_size, words2id, unk, padding):

    # print("load pretrained embedding from {}".format(file))
    # f = open(file, encoding='utf-8')
    # allLines = f.readlines()
    # indexs = set()
    # info = allLines[0].strip().split(' ')
    # embed_dim = len(info) - 1
    # emb = nn.Embedding(vocab_size, embed_dim)
    #
    # # init.uniform(emb.weight, a=-np.sqrt(3 / embed_dim), b=np.sqrt(3 / embed_dim))
    # oov_emb = torch.zeros(1, embed_dim).type(torch.FloatTensor)
    # now_line = 0
    # for line in allLines:
    #     now_line += 1
    #     sys.stdout.write("\rhandling with the {} line.".format(now_line))
    #     info = line.split(" ")
    #     wordID = words2id[info[0]]
    #     if wordID >= 0:
    #         indexs.add(wordID)
    #         for idx in range(embed_dim):
    #             val = float(info[idx + 1])
    #             emb.weight.data[wordID][idx] = val
    #             # oov_emb[0][idx] += val
    # f.close()
    # print("\nhandle finished")
    #
    # unkID = words2id[unk]
    # paddingID = words2id[padding]
    # for idx in range(embed_dim):
    #     emb.weight.data[paddingID][idx] = 0
    #     emb.weight.data[unkID][idx] = 0
    #
    # return emb, embed_dim

    with open(file, encoding="UTF-8") as f:
        allLines = f.readlines()
        indexs = set()
        info = allLines[0].strip().split(' ')
        embDim = len(info) - 1
        emb = nn.Embedding(vocab_size, embDim)
        # init.uniform(emb.weight, a=-np.sqrt(3 / embDim), b=np.sqrt(3 / embDim))
        oov_emb = torch.zeros(1, embDim).type(torch.FloatTensor)

        now_line = 0
        for line in allLines:
            now_line += 1
            sys.stdout.write("\rHandling with the {} line.".format(now_line))
            info = line.split(' ')
            wordID = words2id[info[0]]
            if wordID >= 0:
                indexs.add(wordID)
                for idx in range(embDim):
                    val = float(info[idx + 1])
                    emb.weight.data[wordID][idx] = val
                    oov_emb[0][idx] += val
        f.close()
    print("\nHandle Finished.")
    count = len(indexs) + 1
    for idx in range(embDim):
        oov_emb[0][idx] /= count
    unkID = words2id[unk]
    paddingID = words2id[padding]
    for idx in range(embDim):
        emb.weight.data[paddingID][idx] = 0
    if unkID != -1:
        for idx in range(embDim):
            emb.weight.data[unkID][idx] = oov_emb[0][idx]
    print("Load Embedding file: ", file, ", size: ", embDim)
    oov = 0
    for idx in range(vocab_size):
        if idx not in indexs:
            oov += 1
    print("oov: ", oov, " total: ", vocab_size, "oov ratio: ", oov / vocab_size)
    print("oov ", unk, "use avg value initialize")
    return emb, embDim