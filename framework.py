import numpy as np
import torch
from torch import nn
from utils.util import AverageMeter
# from prefetch_generator import BackgroundGenerator
from utils.util import get_learning_rate, accuracy, record_epoch_learn_alpha, get_fc_name
from models.regularizer import reg_channel_att_fea_map_learn
from models.loss_function import loss_kl, get_fea_map_loss


class TransferFramework:

    def __init__(self, args, train_loader, val_loader, target_class_num, data_aug, base_model_name,
                 model_source, model_feature, model_source_classifier, model_target_classifier, feature_criterions,
                 loss_fn, num_epochs, optimizer, lr_scheduler, writer, logger, print_freq=10):

        self.setting = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_class_num = target_class_num
        self.data_aug = data_aug
        self.reg_type = args.reg_type
        self.feature_criterions = feature_criterions

        self.base_model_name = base_model_name
        self.model_source = model_source

        # target model
        self.model_feature = model_feature
        self.model_source_classifier = model_source_classifier
        self.model_target_classifier = model_target_classifier

        # self.criterion_mse = nn.MSELoss().cuda()
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.optimizer = optimizer

        self.lambada = args.lambada
        self.theta = args.theta

        self.lr = 0.0
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.logger = logger
        self.print_freq = print_freq

        # framework init
        self.hook_layers = []
        if len(self.setting.gpu_id) <= 1:
            self.layer_outputs_source = []
            self.layer_outputs_target = []
        else:
            self.layer_outputs_source = {}
            self.layer_outputs_target = {}
        self.logger.info("hook output save to type: {}".format(type(self.layer_outputs_source)))
        self.framework_init()


    def framework_init(self):
        self.hook_setting()

    # hook
    def _for_hook_source(self, module, input, output):
        if len(self.setting.gpu_id) > 1:
            gpu_id = str(output.get_device())
            if gpu_id not in self.layer_outputs_source:
                self.layer_outputs_source[gpu_id] = []
            self.layer_outputs_source[gpu_id].append(output)
        else:
            self.layer_outputs_source.append(output)

    def _for_hook_target(self, module, input, output):
        if len(self.setting.gpu_id) > 1:
            gpu_id = str(output.get_device())
            if gpu_id not in self.layer_outputs_target:
                self.layer_outputs_target[gpu_id] = []
            self.layer_outputs_target[gpu_id].append(output)
        else:
            self.layer_outputs_target.append(output)

    def register_hook(self, model, func):
        for name, layer in model.named_modules():
            if name in self.hook_layers:
                layer.register_forward_hook(func)


    def get_hook_layers(self):
        if self.setting.base_model_name in ['resnet50']:
            if len(self.setting.gpu_id) > 1:
                self.hook_layers = ['module.layer1.2.conv3', 'module.layer2.3.conv3', 'module.layer3.5.conv3', 'module.layer4.2.conv3']
            else:
                self.hook_layers = ['layer1.2.conv3', 'layer2.3.conv3', 'layer3.5.conv3', 'layer4.2.conv3']
        
        elif self.base_model_name == 'mobilenet_v2':
            if len(self.setting.gpu_id) > 1:
                self.hook_layers = ['module.features.5.conv3', 'module.features.9.conv.3', 'module.features.13.conv.3', 'module.features.17.conv.3']
            else:
                self.hook_layers = ['features.5.conv.3', 'features.9.conv.3', 'features.13.conv.3', 'features.17.conv.3']

        else:
            assert False, self.logger.info("invalid base_model_name={}".format(self.base_model_name))


    def hook_setting(self):
        # hook
        self.get_hook_layers()
        self.register_hook(self.model_source, self._for_hook_source)
        self.register_hook(self.model_feature, self._for_hook_target)
        self.logger.info("self.hook_layers={}".format(self.hook_layers))


    def train(self, epoch):
        # train mode
        # target model
        self.model_feature.train()
        self.model_target_classifier.train()
        self.model_source_classifier.eval()

        # source model
        self.model_source.eval()

        clc_losses = AverageMeter()
        kl_losses = AverageMeter()
        # fm_mse_losses = AverageMeter()
        fea_losses = AverageMeter()

        total_losses = AverageMeter()
        train_top1_accs = AverageMeter()

        self.lr_scheduler.step(epoch)
        self.lr = get_learning_rate(self.optimizer)
        self.logger.info('self.optimizer={}'.format(self.optimizer))

        self.logger.info('kl_loss weight lambada={}'.format(self.lambada))
        self.logger.info('fea_loss weight theta={}'.format(self.theta))
        self.logger.info('T={}'.format(self.setting.T))
        self.logger.info("reg_type: {}".format(self.reg_type))

        for i, (imgs, labels) in enumerate(self.train_loader):

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            # taget forward and loss
            target_outputs = self.model_feature(imgs)

            target_model_source_classifier_outputs = self.model_source_classifier(target_outputs)
            target_model_target_classifier_outputs = self.model_target_classifier(target_outputs)

            # source_model forward for hook
            with torch.no_grad():
                source_outputs = self.model_source(imgs)

            # loss
            clc_loss = self.loss_fn(target_model_target_classifier_outputs, labels)
            kl_loss = loss_kl(target_model_source_classifier_outputs, source_outputs, self.setting.T)
            
            if self.reg_type == 'channel_att_fea_map_learn':
                if self.theta == 0.0:
                    # print('self.theta == 0')
                    fea_loss = 0.0

                else:
                    fea_loss = reg_channel_att_fea_map_learn(self.layer_outputs_source, self.layer_outputs_target,
                                                         self.feature_criterions, self.setting.bits_activations, self.logger)
            else:
                assert False, "Wrong reg type!!!"

            total_loss = clc_loss + self.lambada * kl_loss +  self.theta * fea_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # batch update
            self.layer_outputs_source.clear()
            self.layer_outputs_target.clear()

            clc_losses.update(clc_loss.item(), imgs.size(0))
            kl_losses.update(kl_loss.item(), imgs.size(0))
            # fm_mse_losses.update(fm_mse_loss.item(), imgs.size(0))

            if fea_loss == 0.0:
                fea_losses.update(fea_loss, imgs.size(0))
            else:
                fea_losses.update(fea_loss.item(), imgs.size(0))
            total_losses.update(total_loss.item(), imgs.size(0))

            # compute accuracy
            top1_accuracy = accuracy(target_model_target_classifier_outputs, labels, 1)
            train_top1_accs.update(top1_accuracy, imgs.size(0))

            if i % self.print_freq == 0:
                self.logger.info(
                    'Train Epoch: [{:d}/{:d}][{:d}/{:d}]\tlr={:.6f}\tclc_loss={:.4f}\t\tkl_loss={:.4f}'
                    '\t\tfea_loss={:.4f}\t\ttotal_loss={:.4f}\ttop1_Accuracy={:.4f}'
                        .format(epoch, self.num_epochs, i, len(self.train_loader), self.lr, clc_losses.avg,
                                kl_losses.avg, fea_losses.avg, total_losses.avg, train_top1_accs.avg))

            # break

        # save tensorboard
        self.writer.add_scalar('lr', self.lr, epoch)
        self.writer.add_scalar('Train_classification_loss', clc_losses.avg, epoch)
        self.writer.add_scalar('Train_kl_loss', kl_losses.avg, epoch)
        # self.writer.add_scalar('Train_fm_mse_loss', fm_mse_losses.avg, epoch)
        self.writer.add_scalar('Train_fea_loss', fea_losses.avg, epoch)
        self.writer.add_scalar('Train_total_loss', total_losses.avg, epoch)
        self.writer.add_scalar('Train_top1_accuracy', train_top1_accs.avg, epoch)

        self.logger.info(
            '||==> Train Epoch: [{:d}/{:d}]\tTrain: lr={:.6f}\tclc_loss={:.4f}\t\tkl_loss={:.4f}'
            '\t\tfea_loss={:.4f}\ttotal_loss={:.4f}\ttop1_Accuracy={:.4f}'
                .format(epoch, self.num_epochs, self.lr, clc_losses.avg, kl_losses.avg,
                        fea_losses.avg, total_losses.avg, train_top1_accs.avg))

        return clc_losses.avg, kl_losses.avg, fea_losses.avg, total_losses.avg, train_top1_accs.avg


    def val(self, epoch):
        # test mode
        self.model_feature.eval()
        self.model_target_classifier.eval()

        val_losses = AverageMeter()
        val_top1_accs = AverageMeter()

        # Batches
        for i, (imgs, labels) in enumerate(self.val_loader):
            # Move to GPU, if available
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            if self.data_aug == 'improved':
                bs, ncrops, c, h, w = imgs.size()
                imgs = imgs.view(-1, c, h, w)

            # forward and loss
            with torch.no_grad():
                outputs = self.model_feature(imgs)
                outputs = self.model_target_classifier(outputs)

                if self.data_aug == 'improved':
                    outputs = outputs.view(bs, ncrops, -1).mean(1)

                val_loss = self.loss_fn(outputs, labels)

            val_losses.update(val_loss.item(), imgs.size(0))
            # compute accuracy
            top1_accuracy = accuracy(outputs, labels, 1)
            val_top1_accs.update(top1_accuracy, imgs.size(0))

            # batch update
            self.layer_outputs_source.clear()
            self.layer_outputs_target.clear()

            # Print status
            if i % self.print_freq == 0:
                self.logger.info('Val Epoch: [{:d}/{:d}][{:d}/{:d}]\tval_loss={:.4f}\t\ttop1_accuracy={:.4f}\t'
                            .format(epoch, self.num_epochs, i, len(self.val_loader), val_losses.avg, val_top1_accs.avg))
            # break

        self.writer.add_scalar('Val_loss', val_losses.avg, epoch)
        self.writer.add_scalar('Val_top1_accuracy', val_top1_accs.avg, epoch)

        self.logger.info('||==> Val Epoch: [{:d}/{:d}]\tval_loss={:.4f}\t\ttop1_accuracy={:.4f}'
                         .format(epoch, self.num_epochs, val_losses.avg, val_top1_accs.avg))

        return val_losses.avg, val_top1_accs.avg















