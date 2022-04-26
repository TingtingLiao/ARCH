import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import time
import trimesh
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.model import HGPIFuNet
from lib.data.Evaluator import Evaluator
from lib.common.train_util import *
from lib.common.render import Render
from lib.common.checkpoints import CheckpointIO


class ARCH(torch.nn.Module):
    def __init__(self, cfg):
        super(ARCH, self).__init__()
        self.cfg = cfg
        self.lr_G = cfg.lr_G
        self.ckp_dir = os.path.join(cfg.ckpt_dir, cfg.name)
        self.vis_dir = os.path.join(cfg.results_dir, cfg.name)
        os.makedirs(self.ckp_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        if cfg.log_dir is not None:
            self.log_dir = os.path.join(cfg.log_dir, cfg.name)
            self.logger = SummaryWriter(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)

        self.max_n_iter = cfg.max_n_iter
        self.netG = HGPIFuNet(cfg)
        self.device_ = torch.device(f"cuda:{self.cfg.gpus[0]}")
        self.netG.to(self.device_)

        [self.optimizer_G], [scheduler_G] = self.configure_optimizers()
        self.evaluator = Evaluator(self.device_)
        self.render = Render(device=self.device_)
        self.checkpoint_io = CheckpointIO(self.ckp_dir, model=self.netG, optimizer=self.optimizer_G)
        try:
            load_dict = self.checkpoint_io.load('model.pt')
        except:
            load_dict = dict()
        self.n_iter = load_dict.get('it', 0)

    def render_meshes(self, mesh_list):
        images = []
        for mesh in mesh_list:
            self.render.load_mesh(mesh.vertices, mesh.faces, normalize=True)
            render_geo = self.render.get_clean_image(cam_ids=[0])
            render_geo = render_geo[0][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
            images.append(render_geo * 255)
        images = np.concatenate(images, 1)
        return images

    @staticmethod
    def convert_image_tensor_to_numpy(image_tensor):
        images = []
        for i, im in enumerate(image_tensor):
            im = np.uint8((im.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)[:, :, ::-1] * 255.0)
            images.append(im)
        return np.concatenate(images, 1)

    def vis(self, data, save_path):
        print(save_path)
        mesh_list = gen_mesh(self.cfg, self.netG, self.device_, data, save_path, with_posed_res=True)
        renders = self.render_meshes(mesh_list)
        inputs = self.convert_image_tensor_to_numpy(data['img'])
        vis_image = np.concatenate([inputs, renders], 1)
        cv2.imwrite(save_path[:-4] + '.png', vis_image)

    def configure_optimizers(self):
        weight_decay = self.cfg.weight_decay
        momentum = self.cfg.momentum

        optim_params_G = [{
            'params': self.netG.parameters(),
            'lr': self.lr_G
        }]

        if self.cfg.optim == "Adadelta":
            optimizer_G = torch.optim.Adadelta(optim_params_G,
                                               lr=self.lr_G,
                                               weight_decay=weight_decay)
        elif self.cfg.optim == "Adam":
            optimizer_G = torch.optim.Adam(optim_params_G,
                                           lr=self.lr_G)
        elif self.cfg.optim == "RMSprop":
            optimizer_G = torch.optim.RMSprop(optim_params_G,
                                              lr=self.lr_G,
                                              weight_decay=weight_decay,
                                              momentum=momentum)
        else:
            raise NotImplementedError

        # set scheduler
        scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G, milestones=self.cfg.schedule, gamma=self.cfg.gamma)

        return [optimizer_G], [scheduler_G]

