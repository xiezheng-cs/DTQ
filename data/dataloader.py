#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/23 19:45
# @Author  : xiezheng
# @Site    : 
# @File    : dataloader.py


import os
import sys
import numpy as np

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torchvision.datasets as datasets
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# torchvision.set_image_backend('accimage')


def get_target_dataloader(dataset, batch_size, n_threads, data_path='',  image_size=224,
                          data_aug='default', logger=None):
    """
        Get dataloader for target_dataset
        :param dataset: the name of the dataset
        :param batch_size: how many samples per batch to load
        :param n_threads:  how many subprocesses to use for data loading.
        :param data_path: the path of dataset
        :param logger: logger for logging
    """

    logger.info("|===>Get datalaoder for " + dataset)

    # setting
    crop_size = {299: 320, 224: 256}
    resize = crop_size[image_size]
    logger.info("image_size={}, resize={}".format(image_size, resize))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if data_aug == 'default':
        # torchvision.set_image_backend('accimage')
        # logger.info('torchvision.set_image_backend(\'accimage\')')
        logger.info('data_aug = {} !!!'.format(data_aug))
        train_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            normalize])
        val_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize])

    elif data_aug == 'improved':
        logger.info('data_aug = {} !!!'.format(data_aug))
        train_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(resize),      # important
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            normalize])

        val_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.TenCrop(image_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])

    else:
        assert False, logger.info("invalid data_aug={}".format(data_aug))

    # data root
    if dataset in ['MIT_Indoors_67', 'Stanford_Dogs', 'Caltech_256-10', 'Caltech_256-20',
                   'Caltech_256-30', 'Caltech_256-40', 'Caltech_256-60', 'CUB-200-2011', 'Food-101', 'DeepFashion_0.1',
                   'DeepFashion_0.05']:
        data_root = os.path.join(data_path, dataset)
    else:
        assert False, logger.info("invalid dataset={}".format(dataset))
    logger.info('{} path = {}'.format(dataset, data_root))

    # datset
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'test'), transform=val_transform)
    class_num = len(train_dataset.classes)
    train_dataset_sizes = len(train_dataset)
    val_dataset_sizes = len(val_dataset)

    # dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=n_threads)

    if data_aug == 'improved':
        batch_size = int(batch_size / 4)
        logger.info('{}: batch_size = batch_size / 4 = {}'.format(data_aug, batch_size))

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=n_threads)

    logger.info("train and val loader are ready! class_num={}".format(class_num))
    logger.info("train_dataset_sizes={}, val_dataset_sizes={}".format(train_dataset_sizes, val_dataset_sizes))
    return train_loader, val_loader, class_num, train_dataset_sizes




if __name__ == '__main__':
    print()
