#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/6 10:30
# @Author  : xiezheng
# @Site    : 
# @File    : loss_function.py


import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1), device=outputs.device)
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss



def loss_kl(outputs, teacher_outputs, T=1.0):
    # print('T={}'.format(T))
    kl_loss = (T * T) * nn.KLDivLoss(size_average=False)(F.log_softmax(outputs / T),
                                                         F.softmax(teacher_outputs / T)) / outputs.shape[0]

    return kl_loss


def get_fea_map_loss(layer_outputs_source, layer_outputs_target, criterion_mse):
    fea_loss = torch.tensor(0.).cuda()
    for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
        fea_loss += 0.5 * criterion_mse(fm_tgt, fm_src.detach())
    return fea_loss



def get_loss_type(loss_type, logger=None):
    if loss_type == 'loss_label_smoothing':
        loss_fn = loss_label_smoothing

    elif loss_type == 'CrossEntropyLoss':
        loss_fn =  nn.CrossEntropyLoss().cuda()
    else:
        assert False, logger.info("invalid loss_type={}".format(loss_type))

    if logger is not None:
        logger.info("loss_type={}, {}".format(loss_type, loss_fn))
    return loss_fn



if __name__ == '__main__':
    print()