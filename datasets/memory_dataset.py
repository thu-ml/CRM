'''
Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
'''

import random
import os
import math
import pickle

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from datasets.dataset import Dataset
from util import utils, renderer
from util.utils import DEPTH_MAX, DEPTH_MIN, DEPTH_VALID_MIN, DEPTH_VALID_MAX


class MemoryDataset(Dataset):
    '''
    Torch Dataset for validation and visualization
    '''
    def __init__(self, input_specs, train_specs, glctx, device, validate=False, visual=False):
        # Init
        super(MemoryDataset, self).__init__(device)
        self.glctx = glctx
        self.cam_radius = input_specs['radius']
        self.validate = validate
        self.visual = visual
        self.fovx = np.deg2rad(45)
        self.res = input_specs['resolution']
        self.res_img = input_specs['resolution_img']
        self.res_triview = input_specs.get('resolution_triview')
        self.aspect = self.res[1] / self.res[0]
        self.pts_num = train_specs['sample_points_num']
        # self.query_pts_num = train_specs['query_sample_points_num']
        self.epoch = train_specs['num_epochs']
        self.eva_iter = train_specs['eva_iter']
        self.render_mode = train_specs.get('tex_sup_mode')
        self.dep_sill_mode = train_specs.get('dep&sill_sup_mode')
        self.scale = input_specs['scale']

        self.mesh_dir = input_specs['mesh_dir']
        self.class_name = input_specs['class']
        self.camera_angle_num = input_specs['camera_angle_num']
        self.validate_num = input_specs.get('validate_num')
        self.chunk_num = input_specs.get('chunk_num', 1)
        self.split_dir = input_specs.get('split_dir')
        self.render_img_dir = input_specs.get('render_img_dir', '')
        self.triview_color_dir = input_specs.get('triview_color_dir', '')
        self.triview_xyz_dir = input_specs.get('triview_xyz_dir', '')
        self.prompt_path = input_specs.get('prompt_path', '')
        self.dataset_name = input_specs['dataset_name']

        # if self.split_dir:
        self.train_set = utils.load_item(os.path.join(self.split_dir, 'train.txt'))
        self.val_set = utils.load_item(os.path.join(self.split_dir, 'val.txt'))
        random.seed(2023)
        self.val_list = sorted(list(self.val_set))
        self.val_list = random.sample(self.val_list, k=self.validate_num)
        self.test_set = utils.load_item(os.path.join(self.split_dir, 'test.txt'))
        self.objaverse_exclude = utils.load_item(os.path.join(self.split_dir, 'exclude.txt'))

        self.img_num = input_specs.get('img_num', 24)

        self.uuid2transform = utils.load_transform(input_specs.get('blender_transform'))

        # if self.prompt_path != '':
        self.uuid2prompt = utils.load_prompt(self.prompt_path)

        # if task_mode == "debug":
        #     # load raw vertices and faces
        #     ref_mesh = trimesh.load(self.mesh_dir, force='mesh', process=False)
        #     verts = torch.tensor(ref_mesh.vertices, dtype=torch.float32)
        #     faces = torch.LongTensor(ref_mesh.faces)

        #     if self.pcd_dir.endswith(".usd"):
        #         # load point cloud
        #         pcd = kaolin.io.usd.import_pointclouds(self.pcd_dir)[0]
        #         pcd_tr = pcd.points
        #         pcd_colors = pcd.colors

        #         # scale
        #         verts = verts / torch.max(abs(verts)) * self.scale
        #         pcd_tr = pcd_tr / torch.max(abs(pcd_tr)) * self.scale

        #         self.data = [
        #             {'gt_verts': verts.cpu(), 'gt_faces': faces.cpu(), 'gt_pcd': pcd_tr, 'gt_pcd_colors': pcd_colors}]

        #     elif self.pcd_dir.endswith(".ply") and self.render_img_dir == '':
        #         self.data = self._preprocess(verts, faces, self.pcd_dir)

        #     elif self.pcd_dir.endswith('.ply') and self.render_img_dir != '':
        #         # render imgs
        #         filename = os.path.split(self.render_img_dir)[-1]

        #         if self.dataset_name == 'ShapeNet':
        #             imgdir = os.path.join(self.render_img_dir, 'img', filename, 'models')
        #         elif self.dataset_name == 'Objaverse':
        #             imgdir = self.render_img_dir
        #         else:
        #             raise RuntimeError('Not valid dataset_name')
        #         self.data = self._preprocess(verts, faces, self.pcd_dir, imgdir=imgdir)

        #     else:
        #         raise RuntimeError('Illegal point cloud format')

        # elif task_mode == "batch":
        # multi mesh
        self.data = []
        for class_name in os.listdir(self.mesh_dir):
            if class_name == self.class_name or self.class_name == 'all':
                class_dir = os.path.join(self.mesh_dir, class_name)
                if class_name.endswith('.json'):
                    continue
                for i, object_name in enumerate(tqdm(os.listdir(class_dir))):
                    # if i > 99:
                    #     break
                    # if self.dataset_name == 'ShapeNet' and self.class_name == 'all':
                    #     record = '/'.join([class_name, object_name])
                    # elif self.dataset_name == 'Objaverse':
                    record = os.path.splitext(object_name)[0]
                    # exclude no transform data
                    if record in self.objaverse_exclude:
                        continue
                    # else:
                    #     record = object_name
                    # validate set abort
                    if self.validate and record not in self.val_list:
                        continue
                    # train set abort
                    if not self.validate and record not in self.train_set:
                        continue

                    # load vertices and faces
                    gt_path = os.path.join(class_dir, object_name)

                    ref_mesh = trimesh.load(gt_path, force='mesh', process=False)

                    verts = torch.tensor(ref_mesh.vertices, dtype=torch.float32)
                    faces = torch.LongTensor(ref_mesh.faces)

                    imgdir = os.path.join(self.render_img_dir, record)

                    # skip empty folder
                    if not os.path.exists(imgdir):
                        continue

                    res = self._preprocess(verts, faces, uuid=record, imgdir=imgdir)

                    res.update({'uuid': record})

                    # if self.triview_color_dir != '' and self.triview_xyz_dir != '':
                    img_list, xyz_list = [], []
                    triview_color_dir = os.path.join(self.triview_color_dir, record)
                    triview_xyz_dir = os.path.join(self.triview_xyz_dir, record)
                    for i in range(6):
                        # BGR -> RGB
                        raw = cv2.resize(
                            cv2.imread(os.path.join(triview_color_dir, f'{i:03d}.png'), cv2.IMREAD_UNCHANGED),
                            tuple(self.res_triview))
                        xyz = cv2.resize(
                            cv2.imread(os.path.join(triview_xyz_dir, f'xyz_new_{i:03d}.png'), cv2.IMREAD_UNCHANGED),
                            tuple(self.res_triview))
                        mask = torch.tensor(raw[:, :, [-1]],
                                            dtype=torch.float32).repeat(1, 1, 3) / 255
                        img = torch.tensor(raw[:, :, [2, 1, 0]],
                                            dtype=torch.float32) / 255
                        xyz = torch.tensor(xyz, dtype=torch.float32) / 255

                        img = img * mask
                        xyz = xyz * mask

                        img_list.append(img)
                        xyz_list.append(xyz)

                    triview_imgs = torch.stack(img_list, dim=0)
                    triview_xyzs = torch.stack(xyz_list, dim=0)

                    res.update({
                        'triview_color': triview_imgs,
                        'triview_xyz': triview_xyzs
                    })

                    # if self.prompt_path != '':
                    res.update({
                        'prompt': self.uuid2prompt.get(os.path.splitext(object_name)[0])
                    })

                    # if res is not None:
                    self.data.append(res)

    def _preprocess(self, verts, faces, uuid=None, imgdir=None):
        # rotate
        verts[:, -1] = -verts[:, -1]
        verts = verts[:, [0, 2, 1]]

        transform = self.uuid2transform[uuid]
        transport = torch.tensor(transform['offset'], dtype=torch.float32)
        scale = torch.tensor(transform['scale'], dtype=torch.float32)
        verts = (verts * scale + transport) * self.scale / 0.5

        with open(os.path.join(imgdir, 'meta.pkl'), 'rb') as f:
            k, _, _, _, w2cs = pickle.load(f)

        self.fovx = 2 * math.atan(k[0, 2] / k[0, 0])

        img_list, transform_list = [], []
        depth_list = []
        for i, w2c in enumerate(w2cs):
            img_name = f'{i:03d}.png'

            w2c = torch.tensor(np.stack((w2c[0], -w2c[1], -w2c[2], np.array([0., 0., 0., 1.])), axis=0), dtype=torch.float32)
            c2w = torch.inverse(w2c)
            c2w[:3, 3] *= self.scale / 0.5
            w2c = torch.inverse(c2w)
            # BGR -> RGB -> [-1, 1]
            raw = cv2.resize(
                cv2.imread(os.path.join(imgdir, img_name), cv2.IMREAD_UNCHANGED),
                tuple(self.res_img))
            mask = torch.tensor(raw[:, :, [-1]],
                                dtype=torch.float32).repeat(1, 1, 3) / 255
            img = torch.tensor(raw[:, :, [2, 1, 0]],
                                dtype=torch.float32) / 255

            img = img * mask

            transform_list.append(w2c)
            img_list.append(img)

            # depth
            # syncdreamer
            # if os.path.exists(os.path.join(imgdir, f'{i:03d}-depth.png')):
            depth_name = f'{i:03d}-depth.png'
            dep = cv2.imread(os.path.join(imgdir, depth_name), cv2.IMREAD_UNCHANGED)

            # reference
            # https://github.com/liuyuan-pal/SyncDreamer/blob/232a2ec22f0277b4bd429109fb34eaf8f61b13ad/eval_mesh.py#L16-L22
            dep = dep.astype(np.float32) / 65535 * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
            mask = (dep < DEPTH_VALID_MIN) | (dep > DEPTH_VALID_MAX)

            # # cs
            # elif os.path.exists(os.path.join(imgdir, f'depth_{i:03d}1.png')):
            #     depth_name = f'depth_{i:03d}1.png'
            #     depth_path = os.path.join(imgdir, depth_name)

            #     dep =(np.array(Image.open(depth_path))[:,:,0]- 64) / 127
            #     mask = np.array(Image.open(depth_path))[:,:,0] == 255

            dep = np.where(mask, 0., dep)

            dep = torch.tensor(cv2.resize(dep, tuple(self.res)), dtype=torch.float32) * self.scale / 0.5

            depth_list.append(dep)

            render_imgs = torch.stack(img_list)
            transforms = torch.stack(transform_list)
            render_depths = torch.stack(depth_list).unsqueeze(-1)


        return {
            'verts': verts, 
            'faces': faces,
            'render_imgs': render_imgs, 
            'transforms': transforms, 
            'render_depths': render_depths
            }

    def _sample_scene(self, num, rotate=False):
        batch = 1 if rotate else self.camera_angle_num

        proj_mtx = utils.perspective(self.fovx, self.res[1] / self.res[0])

        # if self.render_mode != 'blender' and self.dep_sill_mode != 'blender':
        #     if rotate:
        #         ang = (idx / 20) * np.pi * 2
        #         mv = utils.translate(0, 0, -self.cam_radius) @ (utils.rotate_x(0.4) @ utils.rotate_y(ang))
        #         mv = mv.repeat(batch, 1, 1)
        #     else:
        #         mv = torch.bmm(
        #             utils.translate(0, 0, -self.cam_radius).repeat(batch, 1, 1),
        #             utils.batch_random_rotation_translation(self.camera_angle_num, 0.25)
        #         )
        #     depths = torch.zeros((batch,) + tuple(self.res) + (1,))
        #     imgs = torch.zeros((batch,) + tuple(self.res) + (3,))
        # else:
        mv = self.data[num].get('transforms')
        img_num = len(mv)
        index = random.sample(range(img_num), k=1 if rotate else self.camera_angle_num)

        mv = mv[index]
        imgs = self.data[num].get('render_imgs')[index]
        depths = self.data[num].get('render_depths')[index]

        mvp = torch.bmm(
            proj_mtx.repeat(batch, 1, 1),
            mv
        )
        # campos = torch.linalg.inv(mv)[:, :3, 3]

        return mv.to(self.device), mvp.to(self.device), imgs.to(self.device), \
            depths.to(self.device), proj_mtx.to(self.device), self.res

    def __len__(self):
        # return self.epoch if self.validate else self.its_per_epc
        return 20 if self.visual else len(self.data)

    def __getitem__(self, idx):
        if self.visual:
            num = idx % len(self.data)
        else:
            num = idx

        # rotate or random scene
        mv, mvp, imgs, depth, persp, iter_res = self._sample_scene(num, rotate=self.visual)

        # if self.dep_sill_mode == 'blender':
        #     alpha = alpha_ctn = (depths > 0).float()
        #     normal, face_normals = renderer.render_mesh(
        #         self.glctx,
        #         self.data[num]['gt_verts'].to(self.device),
        #         self.data[num]['gt_faces'].to(self.device),
        #         mvp, campos, persp, iter_res,
        #         normal_only=True
        #     )
        # else:
        alpha, depth, normal = renderer.render_mesh(
            self.glctx,
            self.data[num]['verts'].to(self.device),
            self.data[num]['faces'].to(self.device),
            mvp, persp, iter_res
        )

        triview_color = self.data[num]['triview_color']
        triview_xyz = self.data[num]['triview_xyz']
        prompt = self.data[num]['prompt']

        res = {
            'uuid': self.data[num].get('uuid'),
            'verts': self.data[num]['verts'],
            'faces': self.data[num]['faces'],
            'render_imgs': imgs[None, ],
            'mv': mv[None, ],
            'mvp': mvp[None, ],
            # 'campos': campos[None, ],
            'resolution': self.res,
            'resolution_img': self.res_img,
            'resolution_triview': self.res_triview,
            'normal': normal[None, ],
            'alpha': alpha[None, ],
            # 'alpha_ctn': alpha_ctn[None, ],
            'depth': depth[None, ],
            'persp': persp,
            'triview_color': triview_color[None, ],
            'triview_xyz': triview_xyz[None, ],
            'prompt': [prompt]
        }
        return res
