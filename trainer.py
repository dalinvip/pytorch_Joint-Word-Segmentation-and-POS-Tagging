# @Author : bamtercelboo
# @Datetime : 2018/8/26 8:30
# @File : trainer.py
# @Last Modify Time : 2018/8/26 8:30
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  trainer.py
    FUNCTION : None
"""

import os
import sys
import time
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
from DataUtils.Optim import Optimizer
from DataUtils.utils import *
from DataUtils.eval import Eval
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    """
        Train
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        Args of data:
            train_iter : train batch data iterator
            dev_iter : dev batch data iterator
            test_iter : test batch data iterator
        Args of train:
            model : nn model
            config : config
        """
        print("Training Start......")
        # for k, v in kwargs.items():
        #     self.__setattr__(k, v)
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.config = kwargs["config"]
        self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        if self.config.learning_algorithm == "SGD":
            self.loss_function = nn.CrossEntropyLoss(reduction="sum")
        else:
            self.loss_function = nn.CrossEntropyLoss(reduction="mean")
        print(self.optimizer)
        self.best_score = Best_Result()
        self.train_eval, self.dev_eval_seg, self.dev_eval_pos, self.test_eval_seg, self.test_eval_pos = Eval(), Eval(), Eval(), Eval(), Eval()
        self.train_iter_len = len(self.train_iter)

    def _clip_model_norm(self, clip_max_norm_use, clip_max_norm):
        """
        :param clip_max_norm_use:  whether to use clip max norm for nn model
        :param clip_max_norm: clip max norm max values [float or None]
        :return:
        """
        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, config, epoch, new_lr):
        """
        :param config:  config
        :param epoch:  epoch
        :param new_lr:  learning rate
        :return:
        """
        if config.use_lr_decay is True and epoch > config.max_patience and (
                epoch - 1) % config.max_patience == 0 and new_lr > config.min_lrate:
            # print("epoch", epoch)
            new_lr = max(new_lr * config.lr_rate_decay, config.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _decay_learning_rate(self, epoch, init_lr):
        """衰减学习率

        Args:
            epoch: int, 迭代次数
            init_lr: 初始学习率
        """
        lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
        # print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def _optimizer_batch_step(self, config, backward_count):
        """
        :return:
        """
        if backward_count % config.backward_batch_size == 0 or backward_count == self.train_iter_len:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch):
        """
        :param epoch:
        :return:
        """
        best_epoch = self.best_score.best_epoch
        if epoch > best_epoch:
            self.best_score.early_current_patience += 1
            print("Dev Has Not Promote {} / {}".format(self.best_score.early_current_patience, self.early_max_patience))
            if self.best_score.early_current_patience >= self.early_max_patience:
                print("Early Stop Train. Best Score Locate on {} Epoch.".format(self.best_score.best_epoch))
                exit()

    def _model2file(self, model, config, epoch):
        """
        :param model:  nn model
        :param config:  config
        :param epoch:  epoch
        :return:
        """
        if config.save_model and config.save_all_model:
            save_model_all(model, config.save_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config.save_best_model_path, config.model_name, self.best_score)
        else:
            print()

    def train(self):
        """
        :return:
        """
        epochs = self.config.epochs
        clip_max_norm_use = self.config.clip_max_norm_use
        clip_max_norm = self.config.clip_max_norm
        new_lr = self.config.learning_rate

        for epoch in range(1, epochs + 1):
            print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, epochs))
            new_lr = self._dynamic_lr(config=self.config, epoch=epoch, new_lr=new_lr)
            # self.optimizer = self._decay_learning_rate(epoch=epoch - 1, init_lr=self.config.learning_rate)
            print("now lr is {}".format(self.optimizer.param_groups[0].get("lr")), end="")
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()
            steps = 1
            backward_count = 0
            self.optimizer.zero_grad()
            back_time = []
            en_time = []
            de_time = []
            loss_time = []
            op_time = []
            acc_time = []
            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                # self.optimizer.zero_grad()
                maxCharSize = batch_features.char_features.size()[1]
                decoder_out, state, e_t, d_t = self.model(batch_features, train=True)
                en_time.append(e_t)
                de_time.append(d_t)
                # acc_s = time.time()
                # self.cal_train_acc(batch_features, self.train_eval, batch_count, decoder_out, maxCharSize, self.config)
                # acc_e = time.time()
                # acc_time.append(acc_e-acc_s)
                loss_s = time.time()
                loss = torch.nn.functional.nll_loss(decoder_out, batch_features.gold_features)
                loss_e = time.time()
                loss_time.append(loss_e-loss_s)
                b_s = time.time()
                loss.backward()
                b_e = time.time()
                back_time.append(b_e - b_s)
                # print(back_time)
                self._clip_model_norm(clip_max_norm_use, clip_max_norm)
                op_s = time.time()
                self._optimizer_batch_step(config=self.config, backward_count=backward_count)
                op_e = time.time()
                op_time.append(op_e - op_s)
                # self.optimizer.step()
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    self.cal_train_acc(batch_features, self.train_eval, batch_count, decoder_out, maxCharSize,
                                       self.config)
                    sys.stdout.write("\nBatch_count = [{}/{}] , Loss is {:.6f} , (Correct/Total_num) = Accuracy ({}/{})"
                                     " = {:.6f}%".format(batch_count + 1, self.train_iter_len, loss.item(),
                                                         self.train_eval.correct_num,
                                                         self.train_eval.gold_num,
                                                         self.train_eval.acc() * 100))
            end_time = time.time()
            # print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
            print("\nBackWord Time {}".format(sum(back_time)))
            print("Encoder Time {}".format(sum(en_time)))
            print("Decoder Time {}".format(sum(de_time)))
            print("Loss Time {}".format(sum(loss_time)))
            print("Optimizer Time {}".format(sum(op_time)))
            print("Cal Acc Time {}".format(sum(acc_time)))
            print("\nTrain Time {:.4f}".format(end_time - start_time))
            self.eval(model=self.model, epoch=epoch, config=self.config)
            self._model2file(model=self.model, config=self.config, epoch=epoch)
            self._early_stop(epoch=epoch)

    def eval(self, model, epoch, config):
        """
        :param model: nn model
        :param epoch:  epoch
        :param config:  config
        :return:
        """
        self.dev_eval_pos.clear()
        self.dev_eval_seg.clear()
        eval_start_time = time.time()
        self.eval_batch(self.dev_iter, model, self.dev_eval_seg, self.dev_eval_pos, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.4f}".format(eval_end_time - eval_start_time))

        self.test_eval_pos.clear()
        self.test_eval_seg.clear()
        eval_start_time = time.time()
        self.eval_batch(self.test_iter, model, self.test_eval_seg, self.test_eval_pos, self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        print("Test Time {:.4f}".format(eval_end_time - eval_start_time))

    def eval_batch(self, data_iter, model, eval_seg, eval_pos, best_score, epoch, config, test=False):
        """
        :param data_iter:  eval data iterator
        :param model:  nn model
        :param eval_seg:  seg eval
        :param eval_pos:  pos eval
        :param best_score:  best score
        :param epoch:  current epoch
        :param config:  config
        :param test:  test
        :return:
        """
        model.eval()
        en_time = []
        de_time = []
        Joint_time = []
        for batch_features in data_iter:
            decoder_out, state, e_t, d_t = model(batch_features, train=False)
            en_time.append(e_t)
            de_time.append(d_t)
            j_s = time.time()
            for i in range(batch_features.batch_length):
                self.jointPRF_Batch(batch_features.inst[i], state.words[i], state.pos_labels[i], eval_seg, eval_pos)
            j_e = time.time()
            Joint_time.append(j_e - j_s)
        print("Encoder Time {}".format(sum(en_time)))
        print("Decoder Time {}".format(sum(de_time)))
        print("Joint Time {}".format(sum(Joint_time)))

        # calculate the F-Score
        seg_p, seg_r, seg_f = eval_seg.getFscore()
        pos_p, pos_r, pos_f = eval_pos.getFscore()

        test_flag = "Test"
        if test is False:
            # print()
            test_flag = "Dev"
            best_score.current_dev_score = pos_f
            if pos_f >= best_score.best_dev_score:
                best_score.best_dev_score = pos_f
                best_score.best_epoch = epoch
                best_score.best_test = True
        if test is True and best_score.best_test is True:
            best_score.p = pos_p
            best_score.r = pos_r
            best_score.f = pos_f

        print(test_flag + " ---->")
        print("seg: precision = {:.4f}%  recall = {:.4f}% , f-score = {:.4f}%".format(seg_p, seg_r, seg_f))
        print("pos: precision = {:.4f}%  recall = {:.4f}% , f-score = {:.4f}%".format(pos_p, pos_r, pos_f))

        if test is True:
            print("The Current Best Dev F-score: {:.4f}%, Locate on {} Epoch.".format(best_score.best_dev_score,
                                                                                      best_score.best_epoch))
        if test is True:
            best_score.best_test = False

    @staticmethod
    def jointPRF_Batch(inst, state_words, state_posLabel, seg_eval, pos_eval):
        """
        :param inst:
        :param state_words:
        :param state_posLabel:
        :param seg_eval:
        :param pos_eval:
        :return:
        """
        words = state_words
        posLabels = state_posLabel
        count = 0
        predict_seg = []
        predict_pos = []

        for idx in range(len(words)):
            w = words[idx]
            posLabel = posLabels[idx]
            predict_seg.append('[' + str(count) + ',' + str(count + len(w)) + ']')
            predict_pos.append('[' + str(count) + ',' + str(count + len(w)) + ']' + posLabel)
            count += len(w)

        seg_eval.gold_num += len(inst.gold_seg)
        seg_eval.predict_num += len(predict_seg)
        for p in predict_seg:
            if p in inst.gold_seg:
                seg_eval.correct_num += 1

        pos_eval.gold_num += len(inst.gold_pos)
        pos_eval.predict_num += len(predict_pos)
        for p in predict_pos:
            if p in inst.gold_pos:
                pos_eval.correct_num += 1

    @staticmethod
    def cal_train_acc(batch_features, train_eval, batch_count, decoder_out, maxCharSize, args):
        """
        :param batch_features:
        :param train_eval:
        :param batch_count:
        :param decoder_out:
        :param maxCharSize:
        :param args:
        :return:
        """
        decoder_out = decoder_out.view(batch_features.batch_length, maxCharSize, decoder_out.size(1))
        label = torch_max(decoder_out)
        train_eval.clear()
        actionID_list = []
        goldID_list = []
        for id_batch in range(batch_features.batch_length):
            inst = batch_features.inst[id_batch]
            train_eval.gold_num += inst.chars_size
            actionID_list.extend(label[id_batch][:inst.chars_size])
            goldID_list.extend(inst.gold_index)
        eq_count = (np.array(actionID_list) - np.array(goldID_list)).tolist().count(0)
        train_eval.correct_num += eq_count

