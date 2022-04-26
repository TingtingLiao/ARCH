import torch.nn as nn

from .geometry import index, orthogonal, perspective


class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        """
        :param projection_mode:
        Either orthogonal or perspective.
        It will call the corresponding function for projection.
        :param error_term:
        nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        """
        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.error_term = error_term

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None

    def forward(self, in_tensor_dict):
        return None

    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        return None

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
        return None

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds[-1]

    def get_error(self, in_tensor_dict):
        '''
        Get the network loss from the last query
        :return: loss term
        '''
        return None
