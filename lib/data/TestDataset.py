 
import trimesh
import os.path as osp
import os
import torch
import glob
import numpy as np
import sys
import random
import human_det
from PIL import ImageFile
import smpl
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from lib.common.render import Render
from lib.common.config import cfg
# for pymaf
from lib.pymaf.models import SMPL, pymaf_net
from lib.pymaf.core import path_config
from lib.pymaf.utils.imutils import process_image
from lib.pymaf.utils.geometry import rotation_matrix_to_angle_axis

from lib.common.config import cfg

SMPL_DIR = cfg.smpl_dir


class TestDataset():
    def __init__(self, cfg, device):
        random.seed(1993)
        self.sub_name = cfg['data_dir'].split('/')[-1]
        self.has_det = cfg['has_det']
        self.hps_type = cfg['hps_type']
        self.image_dir = os.path.join(cfg['data_dir'], 'images')
        self.mask_dir = os.path.join(cfg['data_dir'], 'masks')
        self.device = device
        if self.has_det:
            self.det = human_det.Detection()
        else:
            self.det = None
        keep_lst = sorted(glob.glob(f"{self.image_dir}/*"))
        img_fmts = ['jpg', 'png', 'jpeg', "JPG", 'bmp']
        keep_lst = [
            item for item in keep_lst if item.split(".")[-1] in img_fmts
        ]

        self.subject_list = sorted(
            [item for item in keep_lst if item.split(".")[-1] in img_fmts])

        # random sample 4 images
        idx = np.random.randint(0, len(self.subject_list), cfg['num_views'])
        self.subject_list = [self.subject_list[i] for i in idx]

        if self.hps_type == 'pymaf':
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS).to(self.device)
            self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)['model'], strict=True)
            self.hps.eval()

        # Load SMPL model
        self.smpl_model = SMPL(os.path.join(SMPL_DIR, 'smpl'),
                               batch_size=1,
                               create_transl=False).to(self.device)
        self.render = Render(device=device)

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, index):
        img_path = self.subject_list[index]
        img_name = img_path.split("/")[-1]
        mask_path = os.path.join(self.mask_dir, img_name)
        if not os.path.exists(mask_path):
            mask_path = None
        img_icon, img_hps, img_ori, img_mask, uncrop_param = process_image(img_path, self.det, mask_path)

        data_dict = {
            'name': self.sub_name,
            'img_name': img_name[:-4],
            'image': img_icon.to(self.device).unsqueeze(0),
            'ori_image': img_ori,
            'mask': img_mask,
            'uncrop_param': uncrop_param
        }
        with torch.no_grad():
            preds_dict = self.hps(img_hps.to(self.device))

        data_dict['smpl_faces'] = torch.Tensor(
            self.smpl_model.faces.astype(np.int16)).long().unsqueeze(0).to(self.device)

        if self.hps_type == 'pymaf':
            output = preds_dict['smpl_out'][-1]
            scale, tranX, tranY = output['theta'][0, :3]
            data_dict['betas'] = output['pred_shape']
            data_dict['body_pose'] = output['rotmat'][:, 1:]
            data_dict['global_orient'] = output['rotmat'][:, 0:1]
            data_dict['smpl_verts'] = output['verts']

        elif self.hps_type == 'pare':
            data_dict['body_pose'] = preds_dict['pred_pose'][:, 1:]
            data_dict['global_orient'] = preds_dict['pred_pose'][:, 0:1]
            data_dict['betas'] = preds_dict['pred_shape']
            data_dict['smpl_verts'] = preds_dict['smpl_vertices']
            scale, tranX, tranY = preds_dict['pred_cam'][0, :3]

        trans = torch.tensor([tranX, tranY, 0.0]).to(self.device)
        data_dict['scale'] = scale
        data_dict['trans'] = trans

        return data_dict

