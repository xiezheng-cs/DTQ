#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 20:51
# @Author  : xiezheng
# @Site    : 
# @File    : get_model.py


import pickle
import torch
import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v2
from quantization.qmobilenet import Qmobilenet_v2
from quantization.qresnet import Qresnet50


def pretrained_model_imagenet(base_model):   # load pre-trained model 
    return eval(base_model)(pretrained=True)


def get_base_model(base_model, model_type, logger, args):  # interface for obtaining full precision or low precision model
    if model_type == 'source':
        return pretrained_model_imagenet(base_model)

    elif model_type == 'target':

        if args.bits_weights == 32 and args.bits_activations == 32:   # full precision model
            model_target = pretrained_model_imagenet(base_model)
            logger.info('bits_weights and bits_activations == {}, '
                        'target model is full-precision!'.format(args.bits_weights))
            return model_target

        else:
            if base_model == "mobilenet_v2":
                model_target = Qmobilenet_v2(pretrained=True, bits_weights=args.bits_weights,
                                             bits_activations=args.bits_activations)  # load low-precision mobilenet_v2 model
            elif base_model == "resnet50":
                model_target = Qresnet50(pretrained=True,bits_weights=args.bits_weights,
                                         bits_activations=args.bits_activations)  # load low-precision ResNet-101 model
            else:
                assert False, "The model {} not allowed".format(base_model)

            logger.info('bits_weights and bits_activations == {}, '
                        'target model is low-precision!'.format(args.bits_weights))
            return model_target
    else:
        assert False, "Not exist this model_type {}".format(model_type)


def get_model(base_model_name, base_task, logger, args):  # obtain source and target model
    model_source = get_base_model(base_model_name, "source", logger, args)
    model_target = get_base_model(base_model_name, "target", logger, args)

    logger.info("model_source: {}".format(model_source))
    logger.info("model_target: {}".format(model_target))

    for param in model_source.parameters():
        param.requires_grad = False

    logger.info('base_task = {}, get model_source = {} and model_target ={}'
                .format(base_task, base_model_name, base_model_name))
    return model_source, model_target


def model_split(base_model_name, model, target_class_num, logger, args):  # split the target model into feature extractor and classifier
    if 'resnet' in base_model_name:
        model_source_classifier = model.fc
        logger.info('model_source_classifier:\n{}'.format(model_source_classifier))

        model_target_classifier = nn.Linear(model.fc.in_features, target_class_num)
        logger.info('model_target_classifier:\n{}'.format(model_target_classifier))

        model_feature = model
        model_feature.fc = nn.Identity()
        logger.info('model_feature:\n{}'.format(model_feature))

    elif 'mobilenet' in base_model_name:
        model_source_classifier = model.classifier[1]
        logger.info('model_source_classifier:\n{}'.format(model_source_classifier))

        model_target_classifier = nn.Linear(list(model.classifier.children())[1].in_features,target_class_num)

        logger.info('model_target_classifier:\n{}'.format(model_target_classifier))

        model_feature = model
        model_feature.classifier[1] = nn.Identity()
        logger.info('model_feature:\n{}'.format(model_feature))

    else:
        logger.info('unknown base_model_name={}'.format(base_model_name))

    return model_feature, model_source_classifier, model_target_classifier


if __name__ == '__main__':
    model = resnet50(pretrained=False)
