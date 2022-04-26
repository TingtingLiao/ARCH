# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from yacs.config import CfgNode as CN
import os

_C = CN(new_allowed=True)

# needed by trainer
_C.name = 'default'
_C.gpus = [0]
_C.test_gpus = [0]
_C.root = "../data/"
_C.obj_dir = ''
_C.syn_dir = ''
_C.smpl_dir = './data/smpl_related/models'
_C.ckpt_dir = './results/ckpt/'
_C.results_dir = './results/vis/'
_C.log_dir = './results/logs'
_C.projection_mode = 'orthogonal'
_C.num_views = 1

_C.lr_G = 1e-3
_C.weight_decay = 0.0
_C.momentum = 0.0
_C.optim = 'RMSprop'
_C.schedule = [5, 10, 15]
_C.gamma = 0.1
_C.b_min = [-1, -1.5, -0.5]
_C.b_max = [1., 1., 0.5]

_C.overfit = False
_C.resume = False
_C.test_mode = False
_C.thresh = 0.5
_C.mcube_res = 256
_C.clean_mesh = True

_C.batch_size = 4
_C.num_threads = 8

_C.num_epoch = 10
_C.freq_plot = 0.05
_C.freq_show_train = 2000
_C.freq_show_val = 2000
_C.freq_eval = 0.5
_C.accu_grad_batch = 4
_C.max_n_iter = 100000

_C.net = CN()
_C.net.gtype = 'HGPIFuNet'
_C.net.norm = 'group'
_C.net.norm_mlp = 'group'
_C.net.norm_color = 'group'
_C.net.hg_down = 'ave_pool'
# kernel_size, stride, dilation, padding
_C.net.conv1 = [7, 2, 1, 3]
_C.net.conv3x3 = [3, 1, 1, 1]

_C.net.num_stack = 4
_C.net.num_hourglass = 2
_C.net.hourglass_dim = 256
_C.net.mlp_dim = [280, 1024, 512, 256, 128, 1]
_C.net.res_layers = [2, 3, 4]
_C.net.filter_dim = 256
_C.net.num_joint = 45
_C.net.skip_hourglass = True
_C.net.no_residual = True

_C.dataset = CN()
_C.dataset.name = 'mvphuman'
_C.dataset.root = ''
_C.dataset.scales = [1.0, 100.0, 1.0, 1.0, 100.0 / 39.37]
_C.dataset.th_type = 'train'
_C.dataset.input_size = 512
_C.dataset.num_sample_geo = 10000
_C.dataset.num_sample_color = 0

_C.dataset.sigma_geo = 0.05
_C.dataset.aug = False
_C.dataset.aug_bri = 0.4
_C.dataset.aug_con = 0.4
_C.dataset.aug_sat = 0.4
_C.dataset.aug_hue = 0.4
_C.dataset.aug_blur = 1.0
_C.dataset.random_flip = True
_C.dataset.random_scale = True
_C.dataset.random_trans = True
_C.dataset.add_noise = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`


def update_cfg(cfg_file):
    # cfg = get_cfg_defaults()
    _C.merge_from_file(cfg_file)
    # return cfg.clone()
    return _C


def parse_args(args):
    cfg_file = args.cfg_file
    if args.cfg_file is not None:
        cfg = update_cfg(args.cfg_file)
    else:
        cfg = get_cfg_defaults()

    # if args.misc is not None:
    #     cfg.merge_from_list(args.misc)

    return cfg


def parse_args_extend(args):
    if args.resume:
        if not os.path.exists(args.log_dir):
            raise ValueError(
                'Experiment are set to resume mode, but log directory does not exist.'
            )

        # load log's cfg
        cfg_file = os.path.join(args.log_dir, 'cfg.yaml')
        cfg = update_cfg(cfg_file)

        if args.misc is not None:
            cfg.merge_from_list(args.misc)
    else:
        parse_args(args)
