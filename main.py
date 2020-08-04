import os
import argparse
import json

import random
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from option import Option
from framework import TransferFramework
from models.loss_function import get_loss_type
from data.dataloader import get_target_dataloader
from models.get_model import get_model, model_split
from utils.checkpoint import save_checkpoint, save_model
from models.regularizer import get_feature_criterions, get_reg_criterions
from utils.util import get_logger, output_process, ours_record_epoch_data, write_settings, get_optimier_and_scheduler



def train_net(args, logger, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    logger.info('seed={}'.format(seed))

    # init seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # cudnn.benchmark = True
    cudnn.benchmark = False
    cudnn.deterministic = True  # cudnn

    writer = SummaryWriter(args.outpath)
    start_epoch = 0
    val_best_acc = 0
    val_best_acc_index = 0

    # data_loader
    train_loader, val_loader, target_class_num, dataset_sizes = \
        get_target_dataloader(args.target_dataset, args.batch_size, args.num_workers, args.target_data_dir,
                              image_size=args.image_size, data_aug=args.data_aug, logger=logger)

    # model setting
    model_source, model_target = get_model(args.base_model_name, args.base_task, logger, args)

    # target_model split: (feature, classifier)
    model_feature, model_source_classifier, model_target_classifier = \
        model_split(args.base_model_name, model_target, target_class_num, logger, args)

    if len(args.gpu_id) > 1:
        model_source = nn.DataParallel(model_source)
        model_feature = nn.DataParallel(model_feature)
        model_source_classifier = nn.DataParallel(model_source_classifier)
        model_target_classifier = nn.DataParallel(model_target_classifier)
        model_source = model_source.cuda()
        model_feature = model_feature.cuda()
        model_target_classifier = model_target_classifier.cuda()
        model_source_classifier = model_source_classifier.cuda()
        logger.info("push all model to dataparallel and then gpu")
    else:
        model_source = model_source.cuda()
        model_feature = model_feature.cuda()
        model_target_classifier = model_target_classifier.cuda()
        model_source_classifier = model_source_classifier.cuda()
        logger.info("push all model to gpu")

    # iterations -> epochs
    num_epochs = int(np.round(args.max_iter * args.batch_size / dataset_sizes))
    step = [int(0.67 * num_epochs)]
    logger.info('num_epochs={}, step={}'.format(num_epochs, step))

    # loss
    loss_fn = get_loss_type(loss_type=args.loss_type, logger=logger)

    # get feature_criterions
    if args.reg_type in ['channel_att_fea_map_learn', 'fea_loss']:
        feature_criterions = get_reg_criterions(args, logger)

    # optimizer and lr_scheduler
    optimizer, lr_scheduler = get_optimier_and_scheduler(args, model_feature, model_target_classifier, feature_criterions, 
                                                         step, logger)

    # init framework
    framework = TransferFramework(args, train_loader, val_loader, target_class_num, args.data_aug, args.base_model_name,
                                  model_source, model_feature, model_source_classifier, model_target_classifier,
                                  feature_criterions, loss_fn, num_epochs, optimizer, lr_scheduler,
                                  writer, logger, print_freq=args.print_freq)

    # Epochs
    for epoch in range(start_epoch, num_epochs):
        # train epoch
        clc_loss, kl_loss, fea_loss, train_total_loss, train_top1_acc = framework.train(epoch)
        # val epoch
        val_loss, val_top1_acc = framework.val(epoch)
        # record into txt
        ours_record_epoch_data(args.outpath, epoch, clc_loss, kl_loss, fea_loss, train_total_loss, train_top1_acc, val_loss, val_top1_acc)

        if val_top1_acc >= val_best_acc:
            val_best_acc = val_top1_acc
            val_best_acc_index = epoch
            # save_checkpoint
            save_checkpoint(args.outpath, epoch, model_feature, model_source_classifier, model_target_classifier,
                            optimizer, lr_scheduler, val_best_acc)

        logger.info('||==>Val Epoch: Val_best_acc_index={}\tVal_best_acc={:.4f}\n'.format(val_best_acc_index, val_best_acc))
        # break
    return val_best_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer')
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='the path of config file for training (default: 64)')
    argparses = parser.parse_args()
    args = Option(argparses.conf_path)
    args.set_save_path()

    # args = parse_args()
    best_val_acc_list = []
    logger = None
    temp = args.outpath
    for i in range(1, args.repeat+1):
        if args.repeat != 1:
            args.outpath = temp + "_{:02d}".format(i)

        output_process(args.outpath)
        write_settings(args)
        logger = get_logger(args.outpath, 'attention_transfer_{:02d}'.format(i))
        if i == 1:
            args.copy_code(logger, dst=os.path.join(args.outpath, 'code'))

        val_acc = train_net(args, logger, seed=i)
        best_val_acc_list.append(val_acc)

    acc_mean = np.mean(best_val_acc_list)
    acc_std = np.std(best_val_acc_list)
    for i in range(len(best_val_acc_list)):
        print_str = 'repeat={}\tbest_val_acc={}'.format(i, best_val_acc_list[i])
        logger.info(print_str)
    logger.info('All repeat val_acc_mean={}\tval_acc_std={})'.format(acc_mean, acc_std))
