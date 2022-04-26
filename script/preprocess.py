import os
import sys
import trimesh
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lib.common.mesh_util import query_lbs_weight
from lib.common.config import cfg


def sampling_from_mesh(mesh_file, skin_file, sigma, num_samples, b_min, b_max, cuda):
    mesh = trimesh.load(mesh_file, process=False, maintain_order=True)
    skin_weights = np.load(skin_file)['skin_weight']

    surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * num_samples)
    sample_points = surface_points + np.random.normal(scale=sigma, size=surface_points.shape)
    random_points = np.random.rand(num_samples // 4, 3) * (b_max - b_min) + b_min
    sample_points = torch.from_numpy(np.concatenate([sample_points, random_points], 0)).float()

    weights = query_lbs_weight(sample_points, mesh.vertices, skin_weights, cuda).cpu()

    inside = mesh.ray.contains_points(sample_points)
    inside_points = sample_points[inside]
    inside_weights = weights[inside]

    outside = np.logical_not(inside)
    outside_points = sample_points[outside]
    outside_weights = weights[outside]

    return {
        'positive_points': inside_points,
        'negative_points': outside_points,
        'positive_weights': inside_weights,
        'negative_weights': outside_weights
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=str, help='subject name')
    parser.add_argument('-c', '--config', type=str, default="../configs/arch.yaml")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()

    subject = args.subject
    data_dir = cfg.obj_dir
    sigma = cfg.dataset.sigma_geo
    num_samples = cfg.dataset.num_sample_geo * 10
    b_min = np.array(cfg.b_min)
    b_max = np.array(cfg.b_max)
    save_dir = cfg.sample_dir
    cuda = torch.device('cuda:0')

    mesh_file = os.path.join(data_dir, subject, 'tpose.obj')
    skin_file = os.path.join(data_dir, subject, 'skin_weight.npz')
    data = sampling_from_mesh(
        mesh_file, skin_file, sigma, num_samples, b_min, b_max, cuda
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data, os.path.join(save_dir, f'{subject}.pt'))
    print('Sampling done for %s!' % args.subject)
