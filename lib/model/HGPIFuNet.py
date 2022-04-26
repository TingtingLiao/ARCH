import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import *
from .HGFilters import *
from .net_util import init_net


class HGPIFuNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 cfg,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss()):
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)
        self.opt = cfg.net
        self.num_views = cfg.num_views
        self.image_filter = HGFilter(self.opt)
        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            num_views=self.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.intermediate_preds_list = []

        init_net(self)

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, canon_points, xyz, joints):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param canon_points: [B, 3, N] world space coordinates of points
        :param xyz: : [B, 3, N] correspondence in image space
        :param joints: : [B, 3, J] world space coordinates of joints
        :return: [B, Res, N] predictions for each point
        '''

        (xy, z) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=False).detach().float()

        if self.num_views > 1:
            in_cube = in_cube.view(-1, self.num_views, in_cube.shape[-1])
            in_cube = in_cube.max(1)[0]

        # compute the distance from each point to joints
        if self.opt.num_joint > 0:
            dist = (canon_points[:, :, None, :] - joints[:, :, :self.opt.num_joint, None]) ** 2
            coord_feat = torch.exp(-dist.contiguous().view(dist.shape[0], dist.shape[1] * dist.shape[2], dist.shape[3]))
        else:
            coord_feat = canon_points

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        pred_list = []

        for im_feat in self.im_feat_list:
            point_local_feat_list = [self.index(im_feat, xy), coord_feat]
            if self.opt.skip_hourglass:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)

            # out of image plane is always set to 0
            pred = self.surface_classifier(point_local_feat) * in_cube[:, None]
            pred_list.append(pred)

        self.preds = pred_list

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_preds(self):
        return self.preds[-1]

    def get_error(self, in_tensor_dict):
        error = 0
        for pred in self.preds:
            error += self.error_term(pred, in_tensor_dict['label'])
        error /= len(self.preds)
        return error

    def forward(self, in_tensor_dict):
        self.filter(in_tensor_dict['image'])

        canon_points = in_tensor_dict['canon_points']
        xyz = in_tensor_dict['xyz']
        joints = in_tensor_dict['joints'] if 'joints' in in_tensor_dict else None

        self.query(canon_points, xyz, joints)

        res = self.get_preds()

        error = self.get_error(in_tensor_dict)
        return res, error

