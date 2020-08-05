#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 11:26
# @Author  : xiezheng
# @Site    : 
# @File    : regularizer.py


import numpy as np
import math

import torch
from torch import nn
from utils.util import get_conv_num, get_fc_name, concat_gpu_data
from quantization.google_quantization import quantization_on_input


def flatten_outputs(fea):
    return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))


def get_reg_criterions(args, logger):
    if args.base_model_name in ['resnet50', 'resnet101']:
        in_channels_list = [256, 512, 1024, 2048]
        feature_size = [56, 28, 14, 7]
    elif args.base_model_name in ['inception_v3']:
        in_channels_list = [192, 288, 768, 2048]
        feature_size = [71, 35, 17, 8]
    elif args.base_model_name in ['mobilenet_v2']:
        in_channels_list = [32, 64, 96, 320]
        feature_size = [28, 14, 14, 7]
    else:
        assert False, logger.info('invalid base_model_name={}'.format(args.base_model_name))

    logger.info('in_channels_list={}'.format(in_channels_list))
    logger.info('feature_size={}'.format(feature_size))

    feature_criterions = get_feature_criterions(args, in_channels_list, feature_size, logger)  # obtain channel attentive module
    return feature_criterions


class channel_attention(nn.Module):  # channel attentive module
    def __init__(self, in_channels, feature_size):
        super(channel_attention, self).__init__()

        # channel-wise attention
        self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(in_channels))
        self.softmax = nn.Softmax()

    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)

        # channel-wise attention
        c_f = self.fc1(target_feature_resize)
        c_f = self.relu1(c_f)
        c_f = self.fc2(c_f)
        c_f = c_f.view(b, c)

        # softmax
        channel_attention_weight = self.softmax(c_f + self.bias)  # b*in_channels
        return channel_attention_weight


# obtain channel feature alignment loss
def reg_channel_att_fea_map_learn(layer_outputs_source, layer_outputs_target,
                                  feature_criterions, bits_activations, logger):
    if isinstance(feature_criterions, nn.DataParallel):
        feature_criterions_module = feature_criterions.module
        layer_outputs_source_processed = concat_gpu_data(layer_outputs_source)
        layer_outputs_target_processed = concat_gpu_data(layer_outputs_target)
    else:
        feature_criterions_module = feature_criterions
        layer_outputs_source_processed = layer_outputs_source
        layer_outputs_target_processed = layer_outputs_target

    fea_loss = torch.tensor(0.).cuda()
    for i, (fm_src, fm_tgt, feature_criterion) in \
            enumerate(zip(layer_outputs_source_processed, layer_outputs_target_processed, feature_criterions_module)):
        channel_attention_weight = feature_criterion(fm_src)   # b, c
        b, c, h, w = fm_src.shape

        fm_src = flatten_outputs(fm_src)  # b * c * (hw)
        fm_tgt = flatten_outputs(fm_tgt)

        diff = fm_tgt - fm_src.detach()
        distance = torch.norm(diff, 2, 2)  # b * c
        
        distance = torch.mul(channel_attention_weight, distance ** 2) * c
        fea_loss += 0.5 * torch.sum(distance) / b

    return fea_loss


def get_feature_criterions(args, in_channels_list, feature_size, logger):
    feature_criterions = nn.ModuleList()
    for i in range(len(in_channels_list)):

        if args.reg_type == 'channel_att_fea_map_learn':
            feature_criterions.append(channel_attention(in_channels_list[i], feature_size[i]))
        else:
            assert False, logger.info('invalid reg_type={}'.format(args.reg_type))

    if len(args.gpu_id) <= 1:
        feature_criterions = feature_criterions.cuda()
    else:
        feature_criterions = nn.DataParallel(feature_criterions)
        feature_criterions = feature_criterions.cuda()
    logger.info('feature_criterions={}'.format(feature_criterions))
    return feature_criterions

