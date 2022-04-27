import argparse
import os
import sys
import cv2
import numpy as np
import random
import math
import time
from tqdm import tqdm
import smpl
import pickle as pkl
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lib.renderer.camera import Camera
from lib.renderer.gl.prt_render import PRTRender
from lib.renderer.gl.init_gl import initialize_GL_context
from lib.renderer.mesh import load_obj_mesh, save_obj_mesh, compute_tangent
import lib.renderer.opengl_util as opengl_util
import lib.renderer.prt_util as prt_util
from lib.common.config import cfg
from lib.common.mesh_util import linear_blend_skinning, compute_normal

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=str, help='subject name')
parser.add_argument('-r', '--rotation', type=str, help='rotation num')
args = parser.parse_args()

cfg.freeze()

subject = args.subject
rotation = int(args.rotation)
save_folder = cfg.syn_dir
size = int(cfg.dataset.input_size)

format = 'obj'
scale = 180.0
up_axis = 1
with_light = True
normal = True
animation = True

if os.path.exists(os.path.join(save_folder, subject, 'MASK')):
    if len(os.listdir(os.path.join(save_folder, subject, 'MASK'))) == rotation:
        exit()


canon_file = f'{cfg.obj_dir}/{subject}/tpose.obj'
skin_file = f'{cfg.obj_dir}/{subject}/skin_weight.npz'
tex_file = f'/media/liaotingting/usb/Data/{subject}/indoor/Action03/Action03.jpg'

vertices, faces, textures, face_textures = load_obj_mesh(canon_file, with_texture=True)

if animation:
    # get human motion data
    smpl_model = smpl.create('../data/smpl_related/models')
    motion_file = random.choice('../data/smpl_related/motion/*.pkl')
    motion_data = pkl.load(open(motion_file, 'rb'))
    pose = motion_data['smpl_poses'].reshape(-1, 24, 3)
    pose = torch.from_numpy(random.choice(pose)).float()
    with torch.no_grad():
        output = smpl_model(global_orient=pose[None, :1], body_pose=pose[None, 1:], custom_out=True, return_verts=True)
    jointT = output.joint_transform[0, :24]

    # warp canonical mesh
    skin_weight = torch.from_numpy(np.load(skin_file)['skin_weight']).float()
    vertices = torch.from_numpy(vertices).float()
    vertices = linear_blend_skinning(vertices[None], skin_weight[None], jointT[None])[0]
    vertices = vertices.numpy().astype(np.float64)

normals = compute_normal(vertices, faces)
faces_normals = faces

# render
initialize_GL_context(width=size, height=size)

# center
vmin = vertices.min(0)
vmax = vertices.max(0)
vmed = np.median(vertices, 0)
vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
y_scale = scale / (vmax[up_axis] - vmin[up_axis])

# camera
cam = Camera(width=size, height=size)
cam.ortho_ratio = 0.4 * (512 / size)

prt, face_prt = prt_util.computePRT(vertices, faces, normals, 10, 2)
tan, bitan = compute_tangent(normals)
shs = np.load('env_sh.npy')
rndr = PRTRender(width=size, height=size, ms_rate=16)

# texture
texture_image = cv2.imread(tex_file)
texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
texture_image = cv2.resize(texture_image, (3000, 3000))

rndr.set_norm_mat(y_scale, vmed)
rndr.set_mesh(vertices, faces, normals, faces_normals,
              textures, face_textures,
              prt, face_prt, tan, bitan)
rndr.set_albedo(texture_image)

for y in tqdm(range(0, 360, 360 // rotation)):

    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    R = opengl_util.make_rotate(0, math.radians(y), 0)
    R_B = opengl_util.make_rotate(0, math.radians((y + 180) % 360), 0)

    if up_axis == 2:
        R = np.matmul(R, opengl_util.make_rotate(math.radians(90), 0, 0))

    rndr.rot_matrix = R
    rndr.set_camera(cam)

    dic = {'ortho_ratio': cam.ortho_ratio,
           'scale': y_scale,
           'center': vmed,
           'R': R}

    if with_light:
        # random light
        sh_id = random.randint(0, shs.shape[0] - 1)
        sh = shs[sh_id]
        sh_angle = 0.2 * np.pi * (random.random() - 0.5)
        sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)
        dic.update({"sh": sh})

        rndr.set_sh(sh)
        rndr.analytic = False
        rndr.use_inverse_depth = False

    # ==================================================================

    # calib
    calib = opengl_util.load_calib(dic, render_size=size)

    # export_calib_file = os.path.join(save_folder, subject, 'CALIB', f'{y:03d}.txt')
    # os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
    # np.savetxt(export_calib_file, calib)

    # calib
    dic['calib'] = calib
    os.makedirs(os.path.join(save_folder, subject, 'PARAM'), exist_ok=True)
    # np.save(os.path.join(save_folder, subject, 'PARAM', f'{y:03d}.npy'), dic)
    if animation:
        np.savez(
            os.path.join(save_folder, subject, 'PARAM', f'{y:03d}'),
            ortho_ratio=cam.ortho_ratio,
            scale=y_scale,
            center=vmed,
            R=R,
            pose=pose.numpy(),
            jointT=jointT.numpy()
        )
    else:
        np.savez(
            os.path.join(save_folder, subject, 'PARAM', f'{y:03d}'),
            ortho_ratio=cam.ortho_ratio,
            scale=y_scale,
            center=vmed,
            R=R
        )
    # ==================================================================

    # front render
    rndr.display()
    opengl_util.render_result(rndr, 0, os.path.join(save_folder, subject, 'RENDER', f'{y:03d}.png'))
    opengl_util.render_result(rndr, 2, os.path.join(save_folder, subject, 'MASK', f'{y:03d}.png'))
    if normal:
        opengl_util.render_result(rndr, 1, os.path.join(save_folder, subject, 'NORMAL', f'{y:03d}.png'))


done_jobs = len(os.listdir(save_folder))
all_jobs = len(os.listdir(f"/media/liaotingting/usb3/THuman2.0/data/{dataset}/scans"))
print(
    f"Finish rendering {subject}| {done_jobs}/{all_jobs} | Time: {(time.time() - t0):.0f} secs")
