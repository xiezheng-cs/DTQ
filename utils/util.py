import os
import sys
from datetime import datetime
import numpy as np
import shutil
import json

import torch
import logging
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger


def output_process(output_path):
    if os.path.exists(output_path):
        print("{} file exist!".format(output_path))
        action = input("Select Action: d (delete) / q (quit):").lower().strip()
        act = action
        if act == 'd':
            shutil.rmtree(output_path)
        else:
            raise OSError("Directory {} exits!".format(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr[0]


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.long().view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (1.0 / batch_size)


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.makedirs(folder)


def record_epoch_data(outpath, epoch, clc_loss, mid_clc_loss, train_total_loss, train_top1_acc, val_loss, val_top1_acc):
    txt_path = os.path.join(outpath, "log.txt")
    f = open(txt_path, 'a+')

    record_txt = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n' \
        .format(epoch, clc_loss, mid_clc_loss, train_total_loss, train_top1_acc, val_loss, val_top1_acc)

    if epoch == 0:
        record_head = "epoch\tclc_loss\tmid_clc_loss\t" \
                      "train_total_loss\ttrain_top1_acc\tval_loss\tval_top1_acc\n"
        f.write(record_head)

    f.write(record_txt)
    f.close()


def ours_record_epoch_data(outpath, epoch, clc_loss, kl_loss, fm_mse_loss,
                           train_total_loss, train_top1_acc, val_loss, val_top1_acc):
    txt_path = os.path.join(outpath, "log.txt")
    f = open(txt_path, 'a+')

    record_txt = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n' \
        .format(epoch, clc_loss, kl_loss, fm_mse_loss, train_total_loss, train_top1_acc, val_loss, val_top1_acc)

    if epoch == 0:
        record_head = "epoch\tclc_loss\tkl_loss\tfm_mse_loss\t" \
                      "train_total_loss\ttrain_top1_acc\tval_loss\tval_top1_acc\n"
        f.write(record_head)

    f.write(record_txt)
    f.close()


def record_epoch_learn_alpha(outpath, alpha, epoch, logger):
    txt_path = os.path.join(outpath, "alpha.txt")
    f = open(txt_path, 'a+')
    alpha = alpha.data.cpu().numpy()
    # print(alpha.shape)
    np.savetxt(f, alpha.reshape(1, alpha.shape[0]), fmt='%.6e')
    f.close()
    # assert False
    logger.info("epoch={}, alpha save!".format(epoch))


def write_settings(settings):
    """
    Save expriment settings to a file
    :param settings: the instance of option
    """

    with open(os.path.join(settings.outpath, "settings.log"), "w") as f:
        for k, v in settings.__dict__.items():
            f.write(str(k) + ": " + str(v) + "\n")


def get_optimier_and_scheduler(args, model_feature, model_target_classifier, feature_criterions, step, logger):
    # model_source_classifier fixed
    if len(args.gpu_id) > 1:
        if feature_criterions:
            optimizer = optim.SGD([{'params': model_feature.module.parameters()}, 
                                {'params': model_target_classifier.module.parameters(), 'weight_decay':args.weight_decay},
                                {'params': feature_criterions.module.parameters()}],
                                lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.SGD([{'params': model_feature.module.parameters()},
                                {'params': model_target_classifier.module.parameters(),
                                    'weight_decay':args.weight_decay}],
                                lr=args.lr, momentum=args.momentum)
    else:
        if feature_criterions:
            optimizer = optim.SGD([{'params': model_feature.parameters()}, 
                                {'params': model_target_classifier.parameters(), 'weight_decay':args.weight_decay},
                                {'params': feature_criterions.parameters()}],
                                lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.SGD([{'params': model_feature.parameters()},
                                {'params': model_target_classifier.parameters(),
                                    'weight_decay':args.weight_decay}],
                                lr=args.lr, momentum=args.momentum)

    logger.info('optimizer={}'.format(optimizer))

    # lr_scheduler
    if args.lr_scheduler == 'steplr':
        lr_scheduler = MultiStepLR(optimizer, milestones=step, gamma=args.gamma)
        logger.info('lr_scheduler: SGD MultiStepLR !!!')
    else:
        assert False, logger.info("invalid lr_scheduler={}".format(args.lr_scheduler))

    logger.info('lr_scheduler={}'.format(lr_scheduler))
    return optimizer, lr_scheduler


def get_channel_weight(channel_weight_path, logger=None):
    channel_weights = []
    if channel_weight_path:
        for js in json.load(open(channel_weight_path)):
            js = np.array(js)
            js = (js - np.mean(js)) / np.std(js)  # normalization
            cw = torch.from_numpy(js).float().cuda()
            cw = F.softmax(cw / 5.0).detach()
            channel_weights.append(cw)
    else:
        logger.info("channel_weight_path is None")
        return None

    return channel_weights


def concat_gpu_data(data):  # when data distribute to different GPU, we concat the feature in different GPU but in the same position
    """
    Concat gpu data from different gpu.
    """
    gpu_id = list(data.keys())
    gpu_id.sort()
    main_gpu_id = gpu_id[0]
    data_features = []
    for j, i in enumerate(gpu_id):
        data_Cat = data[i]
        for k, fea in enumerate(data_Cat):
            if j == 0:
                data_features.append(fea)
            else:
                data_features[k] = torch.cat((data_features[k], fea.cuda(int(main_gpu_id))))

    return data_features

def get_conv_num(base_model_name, model_source, fc_name, logger):
    model_source_weights = {}
    if 'resnet' in base_model_name:
        for name, param in model_source.named_parameters():
            # print('name={}'.format(name))
            if not name.startswith(fc_name) and ('conv' in name or 'downsample.0' in name):
                model_source_weights[name] = param.detach()
                logger.info('name={}'.format(name))
    # to do
    # elif 'inception' in base_model_name:
    else:
        assert False, logger.info("invalid base_model_name={}, "
                                  "do not know fc_name ".format(base_model_name))

    layer_length = len(model_source_weights)
    return layer_length


def get_fc_name(base_model_name, logger):
    if 'resnet' in base_model_name:
        fc_name = 'fc.'
    elif 'inception' in base_model_name:
        fc_name = 'fc.'
    else:
        assert False, logger.info("invalid base_model_name={}, "
                                  "do not know fc_name ".format(base_model_name))
    return fc_name


if __name__ == '__main__':
    # 241
    channel_weights_path = './json_result/channel_wei.Stanford_Dogs.json'
    channel_weights = get_channel_weight(channel_weights_path)

    print(len(channel_weights))
    print(channel_weights[0].shape)

    # layer_num = len(channel_weights)
    #
    # layer_index = 1
    # chennel_num = channel_weights[layer_index].shape[0]
    #
    # print(layer_num)
    # print(chennel_num)
    # # print(channel_weights[0])
    # print(channel_weights[layer_index]*chennel_num)
    #
    # result = channel_weights[layer_index] * chennel_num >= 1.0
    # sum_value = torch.sum(result)
    # print(result)
    # print(sum_value)
