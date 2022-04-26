import cv2
import trimesh
import numpy as np
import torch
import os
import sys
from skimage import measure
from lib.model.geometry import orthogonal

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from pytorch3d.ops.knn import knn_points

from lib.common.sdf import create_grid, eval_grid_octree, eval_grid


def reconstruction(net, cuda, resolution, b_min, b_max, calib_tensor,
                   joints, joint_transform,
                   smpl_v, smpl_lbs_weights,
                   use_octree=False, num_samples=10000,
                   thresh=0.5):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param calib_tensor [num_view, 4, 4]
    :param joints [3, N]
    :param joint_transform [num_view, 24, 4, 4]
    :param smpl_v [6890, 3]
    :param smpl_lbs_weights [3, N]

    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :param thresh: threshold for marching cubes
    :return: marching cubes results.
    '''

    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
    joints = joints[None].expand(net.num_views, -1, -1)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        samples = torch.from_numpy(points).to(device=cuda).float()
        # warp canonical points and project to image space
        weights = query_lbs_weight(samples.t(), smpl_v, smpl_lbs_weights)
        weights = weights[None].expand(net.num_views, -1, -1)
        samples = samples[None].expand(net.num_views, -1, -1)
        samples_posed = linear_blend_skinning(samples.transpose(1, 2), weights, joint_transform)
        xyz = orthogonal(samples_posed.transpose(1, 2), calib_tensor)

        net.query(samples, xyz, joints)

        pred = net.get_preds()

        return pred[0][0].detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        normals = normals * 0.5 + 0.5
        return verts, faces[:, [0, 2, 1]], normals, values

    except Exception as e:
        print(e)
        print('error cannot marching cubes')
        # return -1
        exit()


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles,
    # by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal.
    # Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))

    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_ply_mesh_with_color(ply_path, points, color):
    """
    Args:
        ply_path: str to save .ply file
        points: [N, 3]
        color: [N, 3]
    """
    assert points.shape == color.shape and points.shape[1] == 3
    to_save = np.concatenate([points, color], axis=-1)
    np.savetxt(ply_path,
               to_save,
               fmt='%.6f %.6f %.6f %d %d %d',
               comments='',
               header=(
                   'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float '
                   'z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                   points.shape[0])
               )


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()


def mesh_clean(mesh, save_path=None):
    """ clean mesh """
    cc = mesh.split(only_watertight=False)
    out_mesh = cc[0]
    bbox = out_mesh.bounds
    height = bbox[1, 0] - bbox[0, 0]
    for c in cc:
        bbox = c.bounds
        if height < bbox[1, 0] - bbox[0, 0]:
            height = bbox[1, 0] - bbox[0, 0]
            out_mesh = c
    if save_path:
        out_mesh.export(save_path)
    return out_mesh


def linear_blend_skinning(points, weight, G):
    """
    Args:
         points: FloatTensor [batch, N, 3]
         weight: FloatTensor [batch, N, K]
         G: FloatTensor [batch, K, 4, 4]
    Return:
        points_deformed: FloatTensor [batch, N, 3]
    """

    if not weight.shape[0] == G.shape[0]:
        raise AssertionError('batch should be same,', weight.shape, G.shape)
    assert weight.shape[0] == G.shape[0]
    batch = G.size(0)
    T = torch.bmm(weight, G.contiguous().view(batch, -1, 16)).view(batch, -1, 4, 4)
    deformed_points = torch.matmul(T[:, :, :3, :3], points[:, :, :, None])[..., 0] + T[:, :, :3, 3]
    return deformed_points


def query_lbs_weight(points, surface_points, skin_weights, device=None):
    """
        query per vert-to-bone weights from surface
    Args:
        points: FloatTensor [N, 3]
        surface_points: FloatTensor [M, 3]
        skin_weights: FloatTensor [M, J]
        device: torch.device
    return:
        weights: FloatTensor [N, J]
    """
    assert len(points.shape) == 2 and len(surface_points.shape) == 2 and len(skin_weights.shape) == 2
    assert surface_points.shape[0] == skin_weights.shape[0]

    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(surface_points):
        surface_points = torch.as_tensor(surface_points).float()
    if not torch.is_tensor(skin_weights):
        skin_weights = torch.as_tensor(skin_weights).float()

    if device:
        points = points.to(device)
        surface_points = surface_points.to(device)
        skin_weights = skin_weights.to(device)
    _, idx, _ = knn_points(points[None], surface_points[None])
    weights = torch.gather(skin_weights, 0, idx.view(-1, 1).expand(-1, skin_weights.shape[-1]))
    return weights
