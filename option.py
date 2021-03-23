import os
import shutil

from pyhocon import ConfigFactory
from utils.util import is_main_process


class Option(object):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)


        # ------------- data options -------------------------------------------
        # target_dataset
        self.target_dataset = self.conf['target_dataset']    # target dataset name
        self.target_data_dir = self.conf['target_data_dir']  # path for loading data set


        # ------------- general options ----------------------------------------
        self.outpath = self.conf['outpath']   # log path
        self.gpu_id = self.conf['gpu_id']     # GPU id to use, e.g. "0,1,2,3"
        self.seed = self.conf['seed']         # manually set RNG seed
        self.print_freq = self.conf['print_freq']    # print frequency (default: 10)
        self.batch_size = self.conf['batch_size']    # mini-batch size
        self.num_workers = self.conf['num_workers']  # num_workers
        self.exp_id = self.conf['exp_id']            # identifier for experiment


        # ------------- common optimization options ----------------------------
        self.repeat = self.conf['repeat']
        self.lr = float(self.conf['lr'])       # initial learning rate
        self.max_iter = self.conf['max_iter']  # number of total epochs
        self.momentum = self.conf['momentum']  # momentum
        self.weight_decay = float(self.conf['weight_decay'])  # weight decay
        self.gamma = self.conf['gamma']    # the times for drop lr
        self.bits_weights = self.conf['bits_weights']
        self.bits_activations = self.conf['bits_activations']
        self.lambada = self.conf['lambada']  # kl loss weight
        self.theta = self.conf['theta']      # AFA loss weight
        self.T = self.conf['T']   # parameter of the KL loss

        # ------------- model options ------------------------------------------
        self.base_task = self.conf['base_task']
        self.base_model_name = self.conf['base_model_name']
        self.image_size = self.conf['image_size']
        self.data_aug = self.conf['data_aug']
        self.reg_type = self.conf['reg_type']

        self.loss_type = self.conf['loss_type']
        self.lr_scheduler = self.conf['lr_scheduler']


        # ---------- resume or pretrained options ---------------------------------
        # path to pretrained model
        self.pretrain_path = None if len(self.conf['pretrain_path']) == 0 else self.conf['pretrain_path']
        # path to directory containing checkpoint
        self.resume = None if len(self.conf['resume']) == 0 else self.conf['resume']


    def set_save_path(self):
        exp_id = 'log_{}_{}_img{}_da-{}_{}_iter{}_bs{}_{}_lr{}_wd{}_W{}A{}_lambada{}_theta{}_T{}_{}' \
            .format(self.base_task, self.target_dataset, self.image_size, self.data_aug, self.base_model_name,
                    self.max_iter, self.batch_size, self.lr_scheduler, self.lr, self.weight_decay,
                    self.bits_weights, self.bits_activations, self.lambada, self.theta, self.T, self.exp_id)

        path = '{}_{}_da-{}_{}'.format('quantized_transfer', self.target_dataset, self.data_aug, self.base_model_name)
        self.outpath = os.path.join(self.outpath, path, exp_id)
        # self.outpath = os.path.join(self.outpath, exp_id)


    def copy_code(self, logger, src=os.path.abspath("./"), dst="./code/"):
        """
        copy code in current path to a folder
        """
        if is_main_process():
            for f in os.listdir(src):
                if "specific_experiments" in f or "log" in f:
                    continue
                src_file = os.path.join(src, f)
                file_split = f.split(".")
                if len(file_split) >= 2 and file_split[1] == "py":
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    dst_file = os.path.join(dst, f)
                    try:
                        shutil.copyfile(src=src_file, dst=dst_file)
                    except:
                        logger.errro("copy file error! src: {}, dst: {}".format(src_file, dst_file))
                elif os.path.isdir(src_file):
                    deeper_dst = os.path.join(dst, f)
                    self.copy_code(logger, src=src_file, dst=deeper_dst)
