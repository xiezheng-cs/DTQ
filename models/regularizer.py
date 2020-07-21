#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 11:26
# @Author  : xiezheng
# @Site    : 
# @File    : regularizer.py


import numpy as np
# from models.pytorch_ssim import ssim, SSIM
import math

import torch
from torch import nn
from utils.util import get_conv_num, get_fc_name, concat_gpu_data
from quantization.google_quantization import quantization_on_input


# regularization list
def reg_classifier(model, fc_name):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if name.startswith(fc_name):
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls


def reg_l2sp(model, fc_name, model_source_weights):
    fea_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if not name.startswith(fc_name):
            fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
    return fea_loss


def reg_fea_map(layer_outputs_source, layer_outputs_target):

    fea_loss = torch.tensor(0.).cuda()
    for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
        b, c, h, w = fm_src.shape
        fea_loss += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2) / b
    return fea_loss


def flatten_outputs(fea):
    return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))


def reg_att_fea_map(layer_outputs_source, layer_outputs_target, channel_weights):

    fea_loss = torch.tensor(0.).cuda()
    for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source, layer_outputs_target)):
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)
        fm_tgt = flatten_outputs(fm_tgt)
        # div_norm = h * w
        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
        distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
        fea_loss += 0.5 * torch.sum(distance)
    return fea_loss



def get_reg_criterions(args, logger):
    if args.base_model_name in ['resnet50', 'resnet101']:
        in_channels_list = [256, 512, 1024, 2048]
        feature_size = [56, 28, 14, 7]
    elif args.base_model_name in ['inception_v3']:
        # in_channels_list = [64, 192, 288, 768]
        in_channels_list = [192, 288, 768, 2048]
        # to_do
        # feature_size = [147, 71, 35, 17]
        feature_size = [71, 35, 17, 8]
    elif args.base_model_name in ['mobilenet_v2']:
        in_channels_list = [32, 64, 96, 320]

        feature_size = [28, 14, 14, 7]
    else:
        assert False, logger.info('invalid base_model_name={}'.format(args.base_model_name))

    logger.info('in_channels_list={}'.format(in_channels_list))
    logger.info('feature_size={}'.format(feature_size))

    feature_criterions = get_feature_criterions(args, in_channels_list, feature_size, logger)

    return feature_criterions


class pixel_attention(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(pixel_attention, self).__init__()

        # pixel-wise attention
        self.fc1 = nn.Linear(feature_size*feature_size, feature_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, feature_size*feature_size, bias=True)
        self.softmax = nn.Softmax()


    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)

        # pixel-wise attention
        p_f = torch.mean(target_feature_resize, dim=1)     # b * (hw)
        p_f = self.fc1(p_f)
        p_f = self.relu1(p_f)
        p_f = self.fc2(p_f)

        p_f = p_f.view(b, h*w)
        pixel_attention_weight = self.softmax(p_f)
        pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h*w)
        return pixel_attention_weight


class channel_attention(nn.Module):
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


class channel_attention_v1(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(channel_attention_v1, self).__init__()

        # channel-wise attention
        self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, 1, bias=False)
        # self.bias = nn.Parameter(torch.zeros(in_channels))
        self.fc3 = nn.Linear(in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)

        # channel-wise attention
        c_f = self.fc1(target_feature_resize)
        c_f = self.relu1(c_f)
        c_f = self.fc2(c_f)
        c_f = c_f.view(b, c)
        c_f = self.fc3(c_f).squeeze(1)

        # softmax
        # channel_attention_weight = self.softmax(c_f + self.bias)  # b*in_channels
        channel_attention_weight = self.sigmoid(c_f)  # b
        return channel_attention_weight


class channel_attention_v2(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(channel_attention_v2, self).__init__()

        # channel-wise attention
        self.fc1 = nn.Linear(in_channels * feature_size * feature_size, in_channels, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, -1)

        # channel-wise attention
        c_f = self.fc1(target_feature_resize)
        c_f = self.relu1(c_f)
        c_f = self.fc2(c_f).squeeze(1)  # b
        print("before sigmoid: ", c_f)
        # softmax
        # channel_attention_weight = self.softmax(c_f + self.bias)  # b*in_channels
        channel_attention_weight = self.sigmoid(c_f)  # b
        print("after sigmoid: ", channel_attention_weight)
        return channel_attention_weight


class channel_attention_v3(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(channel_attention_v3, self).__init__()

        # channel-wise attention
        # self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(feature_size, 1, bias=False)
        # self.bias = nn.Parameter(torch.zeros(in_channels))
        self.softmax = nn.Softmax()

    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        # target_feature_resize = target_feature.view(b, c, h * w)

        # # channel-wise attention
        # c_f = self.fc1(target_feature_resize)
        # c_f = self.relu1(c_f)
        # c_f = self.fc2(c_f)
        # c_f = c_f.view(b, c)

        # # softmax
        # channel_attention_weight = self.softmax(c_f + self.bias)  # b*in_channels
        c_f = target_feature.new_ones(b, c)
        channel_attention_weight = self.softmax(c_f)
        # print(channel_attention_weight)
        return channel_attention_weight


class channel_pixel_attention(nn.Module):
    def __init__(self, in_channels, feature_size):
        super(channel_pixel_attention, self).__init__()

        # channel-wise attention
        self.fc1 = nn.Linear(feature_size * feature_size, feature_size, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(in_channels))

        # pixel-wise attention
        self.fc3 = nn.Linear(feature_size*feature_size, feature_size, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(feature_size, feature_size*feature_size, bias=True)
        self.softmax = nn.Softmax()


    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)

        # channel-wise attention
        c_f = self.fc1(target_feature_resize)
        c_f = self.relu1(c_f)
        c_f = self.fc2(c_f)
        c_f = c_f.view(b, c)
        channel_attention_weight = self.softmax(c_f + self.bias)    # b*in_channels

        # pixel-wise attention
        p_f = torch.mean(target_feature_resize, dim=1)  # b*(hw)
        # print(p_f.shape)
        p_f = self.fc3(p_f)
        p_f = self.relu3(p_f)
        p_f = self.fc4(p_f)

        p_f = p_f.view(b, h * w)
        pixel_attention_weight = self.softmax(p_f)
        pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h*w)

        return channel_attention_weight, pixel_attention_weight


def reg_pixel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):

    fea_loss = torch.tensor(0.).cuda()
    for i, (fm_src, fm_tgt, feature_criterion) in \
            enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):

        pixel_attention_weight = feature_criterion(fm_src)  # b *1* hw
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)    # b * c * (hw)
        fm_tgt = flatten_outputs(fm_tgt)

        diff = fm_tgt - fm_src.detach()
        diff = torch.mul(pixel_attention_weight, diff) * (h * w)         # b * c * (hw)

        distance = torch.norm(diff, 2, 1)
        distance = distance**2      # b * hw
        fea_loss += 0.5 * torch.sum(distance) / b

    return fea_loss


# def reg_channel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):
#     fea_loss = layer_outputs_target[0].new_zeros(1)
#     for i, (fm_src, fm_tgt, feature_criterion) in \
#             enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):
#         channel_attention_weight = feature_criterion(fm_src)  # b * c
#         b, c, h, w = fm_src.shape
#         fm_src = flatten_outputs(fm_src)  # b * c * (hw)
#         fm_tgt = flatten_outputs(fm_tgt)
#
#         diff = fm_tgt - fm_src.detach()
#         distance = torch.norm(diff, 2, 2)  # b * c
#
#         distance = torch.mul(channel_attention_weight, distance ** 2) * c
#         fea_loss += 0.5 * torch.sum(distance) / b
#
#         # distance = torch.mul(channel_attention_weight, distance ** 2) * c
#         # fea_loss += 0.5 * torch.sum(distance) / (h*w)
#
#         # distance = torch.mul(channel_attention_weight, distance ** 2) * (h * w)
#         # fea_loss += 0.5 * torch.sum(distance) / b
#
#     return fea_loss


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

        # if bits_activations != 32:
        #     fm_src = quantization_on_input(fm_src, bits_activations)

        fm_src = flatten_outputs(fm_src)  # b * c * (hw)
        fm_tgt = flatten_outputs(fm_tgt)

        diff = fm_tgt - fm_src.detach()
        distance = torch.norm(diff, 2, 2)  # b * c

        # distance = torch.mul(channel_attention_weight.unsqueeze(1), distance ** 2) * c
        # distance = torch.mul(channel_attention_weight, distance**2) * c 
        # fea_loss += 0.5 * torch.sum(distance) / b

        # div all
        distance = torch.mul(channel_attention_weight, distance**2) 
        fea_loss += 0.5 * torch.sum(distance) / (b*c*w*h)

        # distance = torch.mul(channel_attention_weight, distance ** 2) * c
        # fea_loss += 0.5 * torch.sum(distance) / (h*w)

        # distance = torch.mul(channel_attention_weight, distance ** 2) * (h * w)
        # fea_loss += 0.5 * torch.sum(distance) / b

    return fea_loss


def reg_channel_pixel_att_fea_map_learn(layer_outputs_source, layer_outputs_target, feature_criterions):

    fea_loss = torch.tensor(0.).cuda()
    for i, (fm_src, fm_tgt, feature_criterion) in \
            enumerate(zip(layer_outputs_source, layer_outputs_target, feature_criterions)):

        channel_attention_weight, pixel_attention_weight = feature_criterion(fm_src)   # b*c, b*1*(hw)
        b, c, h, w = fm_src.shape
        fm_src = flatten_outputs(fm_src)    # b * c * (hw)
        fm_tgt = flatten_outputs(fm_tgt)
        diff = fm_tgt - fm_src.detach()

        # pixel attention
        # diff_weight = torch.mul(1 + pixel_attention_weight, diff)     # b * c * (hw)
        diff_weight = torch.mul(1 + pixel_attention_weight * (h * w), diff)  # b * c * (hw)

        # channel attention
        distance = torch.norm(diff_weight, 2, 2)  # b * c
        # distance = torch.mul(1 + channel_attention_weight, distance ** 2)
        distance = torch.mul(1 + channel_attention_weight * c, distance ** 2)

        fea_loss += 0.5 * torch.sum(distance) / b

    return fea_loss



def get_feature_criterions(args, in_channels_list, feature_size, logger):
    feature_criterions = nn.ModuleList()
    for i in range(len(in_channels_list)):
        if args.reg_type == 'pixel_att_fea_map_learn':
            feature_criterions.append(pixel_attention(in_channels_list[i], feature_size[i]))

        elif args.reg_type == 'channel_att_fea_map_learn':
            feature_criterions.append(channel_attention(in_channels_list[i], feature_size[i]))
        # elif args.reg_type == 'channel_att_fea_map_learn':
        #     feature_criterions.append(channel_attention_v3(in_channels_list[i], feature_size[i]))

        elif args.reg_type == 'channel_pixel_att_fea_map_learn':
            feature_criterions.append(channel_pixel_attention(in_channels_list[i], feature_size[i]))

        # elif args.reg_type == 'channel_att_fea_map_without_params':
        #     feature_criterions.append(channel_attention_without_params())
        #
        # elif args.reg_type == 'pixel_att_fea_map_without_params':
        #     feature_criterions.append(pixel_attention_without_params())

        elif args.reg_type == 'fea_loss':
            return None

        elif args.reg_type == 'finetune':
            pass

        elif args.reg_type == 'l2fe':
            pass
        else:
            assert False, logger.info('invalid reg_type={}'.format(args.reg_type))

    if len(args.gpu_id) <= 1:
        feature_criterions = feature_criterions.cuda()
    else:
        feature_criterions = nn.DataParallel(feature_criterions)
        feature_criterions = feature_criterions.cuda()
    logger.info('feature_criterions={}'.format(feature_criterions))
    return feature_criterions

