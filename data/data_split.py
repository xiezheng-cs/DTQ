#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/13 16:57
# @Author  : xiezheng
# @Site    : 
# @File    : data_split.py


import os
from scipy.io import loadmat
import shutil


def get_path_str(line):
    line = str(line)
    _, path, _ = line.split('\'')
    # print('line={}, path={}'.format(line, path))
    return path


def path_replace(line):
    return line.replace('/', '\\')


def copy_img(root, list, save_path):
    for i in range(list.shape[0]):
        print('i={}'.format(i))
        path = get_path_str(list[i][0])
        source_img_path = path_replace(os.path.join(root, 'Images', path))

        dir_, name = path.split('/')
        target_img_dir = path_replace(os.path.join(save_path, dir_))
        if not os.path.exists(target_img_dir):
            os.makedirs(target_img_dir)

        target_img_path = path_replace(os.path.join(target_img_dir, name))
        print('source_img_path={}, target_img_path={}'.format(source_img_path, target_img_path))
        shutil.copy(source_img_path, target_img_path)


if __name__ == '__main__':
    print()
    root = '\Stanford Dogs 120'

    train_list = loadmat(os.path.join(root, 'train_list.mat'))['file_list']
    save_train_path = '\Stanford Dogs 120\\train'
    copy_img(root, train_list, save_train_path)


    # test_list = loadmat(os.path.join(root, 'test_list.mat'))['file_list']
    # save_test_path = '\Stanford Dogs 120\\test'
    # copy_img(root, test_list, save_test_path)