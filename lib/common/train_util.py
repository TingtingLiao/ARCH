from tqdm import tqdm
import os
import cv2
import trimesh
import numpy as np
from termcolor import colored
from lib.common.mesh_util import *
from lib.model.geometry import *
from lib.common.mesh_util import reconstruction, linear_blend_skinning, query_lbs_weight, mesh_clean


def reshape_multiview_tensors(tensor):
    if isinstance(tensor, list):
        reshaped_tensor = [t.view(t.shape[0] * t.shape[1], * t.shape[2:]) for t in tensor]
        return reshaped_tensor
    elif torch.is_tensor(tensor):
        return tensor.view(tensor.shape[0] * tensor.shape[1], * tensor.shape[2:])
    else:
        raise TypeError('tensor must be a list or toch.Tensor')


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        * sample_tensor.shape[2:]
    )
    return sample_tensor


def gen_mesh(opt, netG, cuda, data, save_path, use_octree=True, with_posed_res=False):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    joints = data['joints'].to(device=cuda)
    smpl_v = data['smpl_v'].to(device=cuda)
    smpl_lbs_weights = data['smpl_lbs_weights'].to(device=cuda)
    joint_transform = data['joint_transform'].to(device=cuda)
    b_min = data["b_min"]
    b_max = data['b_max']

    netG.filter(image_tensor)

    verts, faces, normals, values = reconstruction(
        netG, cuda, opt.mcube_res, b_min, b_max,
        calib_tensor=calib_tensor,
        joints=joints,
        joint_transform=joint_transform,
        smpl_v=smpl_v,
        smpl_lbs_weights=smpl_lbs_weights,
        use_octree=use_octree,
        thresh=opt.thresh
    )

    mesh = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
    if opt.clean_mesh:
        mesh = mesh_clean(mesh)
    mesh.export(save_path)

    res = [mesh]
    if with_posed_res:
        # warp pred canonical mesh by NN Skinning Weights.
        vertices_canon = torch.from_numpy(verts).float().to(device=cuda)
        nn_lbs_weights = query_lbs_weight(vertices_canon, smpl_v, smpl_lbs_weights)
        vertices_posed = linear_blend_skinning(vertices_canon[None].expand(netG.num_views, -1, -1),
                                               nn_lbs_weights[None].expand(netG.num_views, -1, -1),
                                               joint_transform)
        vertices_posed = orthogonal(vertices_posed.transpose(1, 2), calib_tensor).transpose(1, 2)
        vertices_posed = vertices_posed.cpu().numpy()
        vertices_posed[:, :, 1] *= -1
        for i, v_posed in enumerate(vertices_posed):
            mesh = trimesh.Trimesh(v_posed, faces)
            mesh.export(save_path[:-4] + '_posed%d.obj' % i)
            res.append(mesh)
    return res


def adjust_learning_rate(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def calc_error(opt, net, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            samples_posed = data['samples_posed'].to(device=cuda)
            samples_canon = data['samples_canon'][None].to(device=cuda)
            label_tensor = data['labels'][None].to(device=cuda)

            # image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)
            if opt.num_views > 1:
                samples_posed = reshape_sample_tensor(samples_posed)
                samples_canon = reshape_sample_tensor(samples_canon, opt.num_views)

            res, error = net.forward(image_tensor, samples_posed, calib_tensor,
                                     canon_points=samples_canon,
                                     labels=label_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)



def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)
    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def save_samples_rgb(fname, points, rgb):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


def tf_log_convert(log_dict):
    new_log_dict = log_dict.copy()
    for k, v in log_dict.items():
        new_log_dict[k.replace("_", "/")] = v
        del new_log_dict[k]

    return new_log_dict


def bar_log_convert(log_dict, name=None, rot=None):
    from decimal import Decimal

    new_log_dict = {}

    if name is not None:
        new_log_dict['name'] = name[0]
    if rot is not None:
        new_log_dict['rot'] = rot[0]

    for k, v in log_dict.items():
        color = "yellow"
        if 'loss' in k:
            color = "red"
            k = k.replace("loss", "L")
        elif 'acc' in k:
            color = "green"
            k = k.replace("acc", "A")
        elif 'iou' in k:
            color = "green"
            k = k.replace("iou", "I")
        elif 'prec' in k:
            color = "green"
            k = k.replace("prec", "P")
        elif 'recall' in k:
            color = "green"
            k = k.replace("recall", "R")

        if 'lr' not in k:
            new_log_dict[colored(k.split("_")[1],
                                 color)] = colored(f"{v:.3f}", color)
        else:
            new_log_dict[colored(k.split("_")[1],
                                 color)] = colored(f"{Decimal(str(v)):.1E}",
                                                   color)

    if 'loss' in new_log_dict.keys():
        del new_log_dict['loss']

    return new_log_dict


def batch_mean(res, key):
    # recursive mean for multilevel dicts
    return torch.stack([
        x[key] if isinstance(x, dict) else batch_mean(x, key) for x in res
    ]).mean()


def scatter_color_to_image(color, points, image_size=512):
    """
    Args:
        color: FloatTensor of shape [3, N]
        points: FloatTensor of shape [2, N] in [-1， 1]
        image_size: int image resolution
    Returns:
        image [image_size, resolution, 3]
    """
    image = np.ones((image_size, image_size, color.shape[0]))
    xys = ((points[:2] + 1.0) * 0.5 * image_size)

    for xy, c in zip(xys.T, color.T):
        if 0 <= int(xy[0]) < image_size and 0 <= int(xy[1]) < image_size:
            image[int(xy[1]), int(xy[0]), :] = c
    return image