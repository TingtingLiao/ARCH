# This script is borrowed from https://github.com/nkolot/SPIN/blob/master/models/hmr.py

import os
import sys
import cv2
import argparse
import numpy as np
import logging
from termcolor import colored
import torch
from tqdm import tqdm
from skimage import img_as_ubyte
import imageio

torch.backends.cudnn.benchmark = True
sys.path.insert(0, '../../')
logging.getLogger("trimesh").setLevel(logging.ERROR)
from lib.data.TestDataset import TestDataset
from lib.model.geometry import orthogonal, index
from lib.common.config import cfg


def prepare_data(args):
    device = torch.device(f'cuda:{args.gpu_device}')
    dataset = TestDataset(
        {
            'data_dir': args.in_dir,
            'out_dir': args.out_dir,
            'num_views': args.num_views,
            'has_det': True,  # w/ or w/o detection
            'hps_type': 'pymaf',  # pymaf/pare
        }, device)

    print(colored(f"Dataset Size: {len(dataset)}", 'red'))
    pbar = tqdm(dataset)

    optimed_pose = []
    optimed_trans = []
    optimed_betas = []
    optimed_orient = []

    scales = []
    images = []
    for data in pbar:
        pbar.set_description(f"{data['name']}")
        optimed_pose.append(data['body_pose'])  # [1,23,3,3]
        optimed_trans.append(data['trans'])  # [3]
        optimed_betas.append(data['betas'])
        optimed_orient.append(data['global_orient'])
        scales.append(data['scale'])
        images.append(data['image'])

    batch = len(scales)
    in_tensor = {'smpl_faces': data['smpl_faces'], 'image': torch.cat(images)}

    optimed_pose = torch.tensor(torch.cat(optimed_pose),
                                device=device,
                                requires_grad=True)  # [batch,23,3,3]

    optimed_trans = torch.tensor(torch.stack(optimed_trans),
                                 device=device,
                                 requires_grad=True)  # [batch, 3]
    optimed_betas = torch.tensor(torch.stack(optimed_betas).mean(0),
                                 device=device,
                                 requires_grad=True)  # [1,10]
    optimed_orient = torch.tensor(torch.cat(optimed_orient),
                                  device=device,
                                  requires_grad=True)  # [batch,1,3,3]
    scales = torch.tensor(scales, device=device)

    optimizer_smpl = torch.optim.SGD(
        [optimed_pose, optimed_trans, optimed_betas, optimed_orient],
        lr=1e-3,
        momentum=0.9)
    scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_smpl,
        mode='min',
        factor=0.5,
        verbose=0,
        min_lr=1e-5,
        patience=args.patience)

    # smpl optimization
    loop_smpl = tqdm(range(args.loop_smpl))

    out_dir = args.out_dir

    os.makedirs(os.path.join(out_dir, "obj"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "smpl"), exist_ok=True)

    filename_output = os.path.join(out_dir, "smpl", data['name'] + '_smpl.gif')
    writer = imageio.get_writer(filename_output, mode='I', duration=0.05)
    for i in loop_smpl:

        optimizer_smpl.zero_grad()

        # prior_loss, optimed_pose = dataset.vposer_prior(optimed_pose)
        smpl_out = dataset.smpl_model(betas=optimed_betas,
                                      body_pose=optimed_pose,
                                      global_orient=optimed_orient,
                                      custom_out=True,
                                      pose2rot=False)

        smpl_verts = (smpl_out.vertices * scales[:, None, None].expand_as(smpl_out.vertices)) + optimed_trans[:, None]
        smpl_verts *= torch.tensor([1.0, -1.0, -1.0]).to(device)

        visual_frames = []
        smpl_loss = 0
        for idx in range(batch):
            # render silhouette
            dataset.render.load_mesh(smpl_verts[idx], in_tensor['smpl_faces'])
            T_mask_F, T_mask_B = dataset.render.get_silhouette_image()

            # silhouette loss
            bg_color = torch.Tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device)
            gt_arr = ((in_tensor['image'][idx].permute(1, 2, 0) * 0.5 + 0.5 - bg_color).sum(dim=-1) != 0.0).float()
            diff_S = torch.abs(T_mask_F[0] - gt_arr)

            smpl_loss += diff_S.mean()

            visual_frame = torch.cat([
                in_tensor['image'][idx],
                diff_S[:, :512].unsqueeze(0).repeat(3, 1, 1)], 1).permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
            visual_frames.append(visual_frame)

        loop_smpl.set_description(f"Body Fitting = {smpl_loss:.3f}")

        # save to gif file
        visual_frames = np.concatenate(visual_frames, 1)
        writer.append_data(img_as_ubyte(visual_frames))

        if i in [0, args.loop_smpl - 1]:
            cv2.imwrite(os.path.join(out_dir, "smpl", 'iter' + str(i) + '.png'), visual_frames[..., ::-1] * 255)

        smpl_loss.backward(retain_graph=True)
        optimizer_smpl.step()
        scheduler_smpl.step(smpl_loss)
        in_tensor['smpl_verts'] = smpl_verts

    calib = torch.stack([
        torch.FloatTensor(
            [
                [scale, 0, 0, trans[0]],
                [0, scale, 0, trans[1]],
                [0, 0, -scale, -trans[2]],
                [0, 0, 0, 1],
            ]
        ) for trans, scale in zip(optimed_trans, scales)])

    # ----------------
    # get smpl parameter
    with torch.no_grad():
        smpl_model = smpl.create('./data/smpl_related/models')
        output = smpl_model(
            betas=optimed_betas.detach().cpu(),
            global_orient=optimed_orient.detach().cpu(),
            body_pose=optimed_pose.detach().cpu(),
            pose2rot=False, custom_out=True)
        joint_transform = output.joint_transform[:, :24]

        debug = True
        if debug:
            vertices = orthogonal(output.vertices.transpose(1, 2).float().cpu(), calib).detach()
            color = index(in_tensor['image'][:, [2, 1, 0]].cpu() * 0.5 + 0.5, vertices[:, :2])
            render_size = 512
            pts = 0.5 * (vertices + 1.0) * 512

            im_list = []
            for i in range(pts.shape[0]):
                im = np.ones((512, 512, 3)) * 0.5
                for p, c in zip(pts[i].T, color[i].T):
                    if 0 < p[0] < render_size and 0 < p[1] < render_size:
                        im[int(p[1]), int(p[0])] = c
                im_list.append(im)
            im_list = np.concatenate(im_list, 1)
            cv2.imwrite(os.path.join(out_dir, 'smpl', 'project.png'), im_list * 255)

            im = (in_tensor['image'][0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5)
            cv2.imwrite(os.path.join(out_dir, 'smpl', 'inputs.png'), im[:, :, ::-1] * 255)

            output = smpl_model(betas=optimed_betas.detach().cpu(), custom_out=True)

    np.savez(
        os.path.join(out_dir, 'smpl', 'param'),
        name=data['name'],
        calib=calib.numpy(),
        joint_transform=joint_transform.numpy(),
        smpl_v=output.vertices[0].numpy(),
        smpl_lbs_weights=smpl_model.lbs_weights.numpy(),
        joints=output.joints[0, :24].numpy(),
        betas=optimed_betas.detach().cpu().numpy(),
        global_orient=optimed_orient.detach().cpu().numpy(),
        body_pose=optimed_pose.detach().cpu().numpy(),
    )

    data = {
        'name': data['name'],
        'img': in_tensor['image'],
        'calib': calib,
        'joints': output.joints[0].t(),
        'joint_transform': joint_transform,
        'smpl_v': output.vertices[0],
        'smpl_lbs_weights': smpl_model.lbs_weights,
    }

    return data


if __name__ == '__main__':
    import smpl
    from .arch import ARCH

    # python -m apps.infer -cfg ./configs/arch.yaml -gpu 0 -in_dir ./examples -out_dir ./results
    # loading cfg file
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu_device', type=int, default=1)
    parser.add_argument('-loop_smpl', '--loop_smpl', type=int, default=100)
    parser.add_argument('-patience', '--patience', type=int, default=5)
    parser.add_argument('-in_dir', '--in_dir', type=str, default="./examples/232932")
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./examples/232932")
    parser.add_argument('-cfg', '--config', type=str, default="configs/arch.yaml")
    parser.add_argument('-nv', '--num_views', type=int, default=1)
    args = parser.parse_args()

    # cfg read and merge
    cfg.merge_from_file(args.config)
    cfg.merge_from_file('./lib/pymaf/configs/pymaf_config.yaml')

    cfg_show_list = [
        'test_gpus', [args.gpu_device],
        'num_views', args.num_views,
        'mcube_res', 512,
        'clean_mesh', False,
        'log_dir', None,
    ]
    cfg.merge_from_list(cfg_show_list)
    cfg.freeze()

    model = ARCH(cfg)

    save_path = os.path.join(args.out_dir, 'obj', 'mesh.obj')
    data = prepare_data(args)

    data.update(
        {
            'b_min': np.array(cfg.b_min),
            'b_max': np.array(cfg.b_max),
        }
    )

    model.vis(data, save_path)
