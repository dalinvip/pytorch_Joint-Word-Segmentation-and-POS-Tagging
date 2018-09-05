# @Author : bamtercelboo
# @Datetime : 2018/8/24 9:58
# @File : utils.py
# @Last Modify Time : 2018/8/24 9:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  utils.py
    FUNCTION : None
"""
import sys
import os
import torch
import numpy as np


class Best_Result:
    """
        Best_Result
    """
    def __init__(self):
        self.current_dev_score = -1
        self.best_dev_score = -1
        self.best_score = -1
        self.best_epoch = 1
        self.best_test = False
        self.early_current_patience = 0
        self.p = -1
        self.r = -1
        self.f = -1


def getMaxindex(model_out, label_size, args):
    """
    :param model_out: model output
    :param label_size: label size
    :param args: argument
    :return: max index for predict
    """
    max = model_out.data[0]
    maxIndex = 0
    for idx in range(1, label_size):
        if model_out.data[idx] > max:
            max = model_out.data[idx]
            maxIndex = idx
    return maxIndex


def getMaxindex_np(model_out):
    """
    :param model_out: model output
    :return: max index for predict
    """
    model_out_list = model_out.data.tolist()
    maxIndex = model_out_list.index(np.max(model_out_list))
    return maxIndex


def getMaxindex_batch(model_out):
    """
    :param model_out: model output
    :return: max index for predict
    """
    model_out_list = model_out.data.tolist()
    maxIndex_batch = []
    for l in model_out_list:
        maxIndex_batch.append(l.index(np.max(l)))

    return maxIndex_batch


def save_model_all(model, save_dir, model_name, epoch):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param model_name:  model name
    :param epoch:  epoch
    :return:  None
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    # torch.save(model.state_dict(), save_path)
    output.close()


def save_best_model(model, save_dir, model_name, best_eval):
    """
    :param model:  nn model
    :param save_dir:  save model direction
    :param model_name:  model name
    :param best_eval:  eval best
    :return:  None
    """
    if best_eval.current_dev_score >= best_eval.best_dev_score:
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        model_name = "{}.pt".format(model_name)
        save_path = os.path.join(save_dir, model_name)
        print("save best model to {}".format(save_path))
        # if os.path.exists(save_path):  os.remove(save_path)
        output = open(save_path, mode="wb")
        torch.save(model.state_dict(), output)
        # torch.save(model.state_dict(), save_path)
        output.close()
        best_eval.early_current_patience = 0


# adjust lr
def get_lrate(optim):
    """
    :param optim: optimizer
    :return:
    """
    for group in optim.param_groups:
        yield group['lr']


def set_lrate(optim, lr):
    """
    :param optim:  optimizer
    :param lr:  learning rate
    :return:
    """
    for group in optim.param_groups:
        group['lr'] = lr

