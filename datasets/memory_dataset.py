# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.
import json
import random
import time
import cv2
import pickle

import numpy as np
import torch
import trimesh
import kaolin
import os
import math
from tqdm import tqdm
from PIL import Image

from .dataset import Dataset
from util import utils, renderer
from util.utils import DEPTH_MAX, DEPTH_MIN, DEPTH_VALID_MIN, DEPTH_VALID_MAX


class MemoryDataset(Dataset):
    def __init__(self, input_specs, train_specs, glctx, tet_grid_size, device, task_mode="debug", validate=False, visual=False):
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
        self.query_pts_num = train_specs['query_sample_points_num']
        self.epoch = train_specs['num_epochs']
        self.eva_iter = train_specs['eva_iter']
        self.render_mode = train_specs.get('tex_sup_mode')
        self.dep_sill_mode = train_specs.get('dep&sill_sup_mode')
        self.scale = input_specs['scale']

        # self.pcd_dir = input_specs['pcd_dir']
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

        if self.split_dir:
            self.train_set = utils.load_item(os.path.join(self.split_dir, 'train.txt'))
            self.val_set = utils.load_item(os.path.join(self.split_dir, 'val.txt'))
            random.seed(2023)
            self.val_list = sorted(list(self.val_set))
            self.val_list = random.sample(self.val_list, k=self.validate_num)
            self.test_set = utils.load_item(os.path.join(self.split_dir, 'test.txt'))
            self.objaverse_exclude = utils.load_item(os.path.join(self.split_dir, 'exclude.txt'))

        self.img_num = input_specs.get('img_num', 24)

        self.uuid2transform = utils.load_transform(input_specs.get('blender_transform'))

        if self.prompt_path != '':
            self.uuid2prompt = utils.load_prompt(self.prompt_path)

        if task_mode == "debug":
            # load raw vertices and faces
            ref_mesh = trimesh.load(self.mesh_dir, force='mesh', process=False)
            verts = torch.tensor(ref_mesh.vertices, dtype=torch.float32)
            faces = torch.LongTensor(ref_mesh.faces)

            if self.pcd_dir.endswith(".usd"):
                # load point cloud
                pcd = kaolin.io.usd.import_pointclouds(self.pcd_dir)[0]
                pcd_tr = pcd.points
                pcd_colors = pcd.colors

                # scale
                verts = verts / torch.max(abs(verts)) * self.scale
                pcd_tr = pcd_tr / torch.max(abs(pcd_tr)) * self.scale

                self.data = [
                    {'gt_verts': verts.cpu(), 'gt_faces': faces.cpu(), 'gt_pcd': pcd_tr, 'gt_pcd_colors': pcd_colors}]

            elif self.pcd_dir.endswith(".ply") and self.render_img_dir == '':
                self.data = self._preprocess(verts, faces, self.pcd_dir)

            elif self.pcd_dir.endswith('.ply') and self.render_img_dir != '':
                # render imgs
                filename = os.path.split(self.render_img_dir)[-1]

                if self.dataset_name == 'ShapeNet':
                    imgdir = os.path.join(self.render_img_dir, 'img', filename, 'models')
                elif self.dataset_name == 'Objaverse':
                    imgdir = self.render_img_dir
                else:
                    raise RuntimeError('Not valid dataset_name')
                self.data = self._preprocess(verts, faces, self.pcd_dir, imgdir=imgdir)

            else:
                raise RuntimeError('Illegal point cloud format')

        elif task_mode == "batch":
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
                        if self.dataset_name == 'ShapeNet' and self.class_name == 'all':
                            record = '/'.join([class_name, object_name])
                        elif self.dataset_name == 'Objaverse':
                            record = os.path.splitext(object_name)[0]
                            # exclude no transform data
                            if record in self.objaverse_exclude:
                                continue
                        else:
                            record = object_name
                        # validate set abort
                        if self.validate and record not in self.val_list:
                            continue
                        # train set abort
                        if not self.validate and record not in self.train_set:
                            continue

                        # if self.dataset_name == 'ShapeNet':
                        #     # load vertices and faces
                        #     gt_path = os.path.join(class_dir, object_name, 'blank_model.ply')
                        #     # load pcd
                        #     pcd_file = os.path.join(self.pcd_dir, class_name, object_name,
                        #                             'models',
                        #                             f'pc_model_normalized.obj_mat_{self.img_num}_262144_131072.ply')

                        if self.dataset_name == 'Objaverse':
                            # load vertices and faces
                            gt_path = os.path.join(class_dir, object_name)
                            # pcd_file = os.path.join(self.pcd_dir, record, 'pc_262144_131072.ply')

                        # skip empty folder
                        if not os.path.exists(gt_path):
                            continue

                        # # skip empty folder
                        # if not os.path.exists(pcd_file):
                        #     continue

                        # load raw vertices and faces
                        try:
                            ref_mesh = trimesh.load(gt_path, force='mesh', process=False)
                        except Exception as e:
                            continue

                        if not hasattr(ref_mesh, 'vertices') or not hasattr(ref_mesh, 'faces'):
                            continue

                        # # load point cloud
                        # pcd = trimesh.load(pcd_file)

                        # if not hasattr(pcd, 'vertices') or not hasattr(pcd, 'colors'):
                        #     print("pcd has no attr vertices or colors", pcd_file)
                        #     continue

                        # if len(pcd.vertices) < self.query_pts_num:
                        #     print("pcd num less than query_pts_num", pcd_file)
                        #     continue

                        verts = torch.tensor(ref_mesh.vertices, dtype=torch.float32)
                        faces = torch.LongTensor(ref_mesh.faces)

                        if self.render_img_dir == '':
                            pass
                            # res = self._preprocess(verts, faces, pcd_file, uuid=record)
                        else:
                            # if self.dataset_name == 'ShapeNet':
                            #     imgdir = os.path.join(self.render_img_dir, class_name, object_name, 'img', object_name, 'models')
                            if self.dataset_name == 'Objaverse':
                                imgdir = os.path.join(self.render_img_dir, record)
                            else:
                                raise RuntimeError('Not valid dataset_name')
                            # skip empty folder
                            if not os.path.exists(imgdir):
                                continue
                            res = self._preprocess(verts, faces, uuid=record, imgdir=imgdir)

                        res[0].update({'uuid': record})

                        if self.triview_color_dir != '' and self.triview_xyz_dir != '':
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

                            res[0].update({
                                'triview_color': triview_imgs,
                                'triview_xyz': triview_xyzs
                            })

                        if self.prompt_path != '':
                            res[0].update({
                                'prompt': self.uuid2prompt.get(os.path.splitext(object_name)[0])
                            })

                        if res is not None:
                            self.data.extend(res)

        else:
            raise ValueError("Error! Unrecognized task mode.")

    def _preprocess(self, verts, faces, uuid=None, imgdir=None):
        # no render images or no blender mode
        if imgdir is None or (self.render_mode != 'blender' and self.dep_sill_mode != 'blender'):
            pass
        #     verts -= ((verts.max(0)[0] + verts.min(0)[0]) / 2).unsqueeze(0).repeat(
        #         (verts.shape[0], 1))

        #     # load point cloud
        #     pcd = trimesh.load(pcd_file)
        #     pcd_tr = torch.tensor(pcd.vertices, dtype=torch.float32)

        #     # rotate & transport
        #     pcd_tr = pcd_tr @ utils.rotate_x(-math.pi / 2).T[:3, :3]
        #     pcd_tr -= ((pcd_tr.max(0)[0] + pcd_tr.min(0)[0]) / 2).unsqueeze(0).repeat((pcd_tr.shape[0], 1))

        #     # color, normalized to [-1, 1]
        #     pcd_colors = ((2 * torch.tensor(pcd.colors, dtype=torch.float32) - 255) / 255)[..., :3]

        #     # scale
        #     verts = verts / torch.max(abs(verts)) * self.scale
        #     pcd_tr = pcd_tr / torch.max(abs(pcd_tr)) * self.scale

        #     return [{
        #             'gt_verts': verts,
        #             'gt_faces': faces,
        #             'gt_pcd': pcd_tr,
        #             'gt_pcd_colors': pcd_colors
        #         }]

        else:
        # # load point cloud
        # pcd = trimesh.load(pcd_file)

        # if not hasattr(pcd, 'vertices') or not hasattr(pcd, 'colors'):
        #     return

        # if len(pcd.vertices) < self.query_pts_num:
        #     return

        # pcd_tr = torch.tensor(pcd.vertices, dtype=torch.float32)

        # rotate
            verts[:, -1] = -verts[:, -1]
            verts = verts[:, [0, 2, 1]]
        # pcd_colors = ((2 * torch.tensor(pcd.colors, dtype=torch.float32) - 255) / 255)[..., :3]

        # if self.dataset_name == 'ShapeNet':
        #     verts *= self.scale / 0.5
        #     pcd_tr *= self.scale / 0.5

        #     with open(os.path.join(imgdir, 'transforms.json'), 'r') as f:
        #         metadata = json.load(f)
        #     self.fovx = metadata['camera_angle_x']
        #     img_list, transform_list = [], []
        #     depth_list = []
        #     for i, data in enumerate(metadata['frames']):
        #         img_name = data['file_path']
        #         assert i == int(os.path.splitext(img_name)[0])

        #         c2w = torch.tensor(data['transform_matrix'], dtype=torch.float32)

        #         c2w[:3, 3] *= self.scale / 0.5

        #         w2c = torch.inverse(c2w)
        #         # BGR -> RGB -> [-1, 1]
        #         raw = cv2.resize(
        #             cv2.imread(os.path.join(imgdir, img_name), cv2.IMREAD_UNCHANGED),
        #             tuple(self.res_img))
        #         mask = torch.tensor(raw[:, :, [-1]],
        #                             dtype=torch.float32).repeat(1, 1, 3) / 255
        #         img = torch.tensor(raw[:, :, [2, 1, 0]],
        #                             dtype=torch.float32) / 255

        #         img = img * mask

        #         transform_list.append(w2c)
        #         img_list.append(img)

        #         # depth
        #         depth_name = f'{i:05d}_depth.png'
        #         dep = cv2.imread(os.path.join(imgdir, depth_name), cv2.IMREAD_UNCHANGED)

        #         # reference
        #         # https://github.com/openai/shap-e/blob/50131012ee11c9d2617f3886c10f000d3c7a3b43/shap_e/rendering/blender/view_data.py#L57-L62
        #         inf_dist = dep == 0xFFFF
        #         dep = np.where(
        #             inf_dist,
        #             0.,
        #             SHAPENET_MAX_DEPTH * (dep.astype(np.float32) / 65536),
        #         )

        #         dep = torch.tensor(cv2.resize(dep, tuple(self.res)), dtype=torch.float32) * self.scale / 0.5

        #         depth_list.append(dep)

        #     gt_render_imgs = torch.stack(img_list)
        #     gt_transforms = torch.stack(transform_list)
        #     gt_render_depths = torch.stack(depth_list).unsqueeze(-1)

            if self.dataset_name == 'Objaverse':
                if self.uuid2transform:
                    transform = self.uuid2transform[uuid]
                    transport = torch.tensor(transform['offset'], dtype=torch.float32)
                    scale = torch.tensor(transform['scale'], dtype=torch.float32)
                    verts = (verts * scale + transport) * self.scale / 0.5
                else:
                    transport = (verts.max(0)[0] + verts.min(0)[0]) / 2
                    verts -= transport
                    scale = 1 / max(verts.max(0)[0] - verts.min(0)[0])

                    verts *= self.scale / 0.5 * scale
                # pcd_tr *= self.scale / 0.5

                with open(os.path.join(imgdir, 'meta.pkl'), 'rb') as f:
                    K, _, _, _, w2cs = pickle.load(f)

                self.fovx = 2 * math.atan(K[0, 2] / K[0, 0])

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
                    if os.path.exists(os.path.join(imgdir, f'{i:03d}-depth.png')):
                        depth_name = f'{i:03d}-depth.png'
                        dep = cv2.imread(os.path.join(imgdir, depth_name), cv2.IMREAD_UNCHANGED)

                        # reference
                        # https://github.com/liuyuan-pal/SyncDreamer/blob/232a2ec22f0277b4bd429109fb34eaf8f61b13ad/eval_mesh.py#L16-L22
                        dep = dep.astype(np.float32) / 65535 * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
                        mask = (dep < DEPTH_VALID_MIN) | (dep > DEPTH_VALID_MAX)

                    # cs
                    elif os.path.exists(os.path.join(imgdir, f'depth_{i:03d}1.png')):
                        depth_name = f'depth_{i:03d}1.png'
                        depth_path = os.path.join(imgdir, depth_name)

                        dep =(np.array(Image.open(depth_path))[:,:,0]- 64) / 127
                        mask = np.array(Image.open(depth_path))[:,:,0] == 255

                    dep = np.where(mask, 0., dep)

                    dep = torch.tensor(cv2.resize(dep, tuple(self.res)), dtype=torch.float32) * self.scale / 0.5

                    depth_list.append(dep)

                gt_render_imgs = torch.stack(img_list)
                gt_transforms = torch.stack(transform_list)
                gt_render_depths = torch.stack(depth_list).unsqueeze(-1)


            return [{
                'gt_verts': verts.cpu(), 'gt_faces': faces.cpu(),
                # 'gt_pcd': pcd_tr, 'gt_pcd_colors': pcd_colors,
                'gt_render_imgs': gt_render_imgs, 'gt_transforms': gt_transforms, 'gt_render_depths': gt_render_depths}]

    def _sample_scene(self, idx, num, rotate=False):
        batch = 1 if rotate else self.camera_angle_num

        proj_mtx = utils.perspective(self.fovx, self.res[1] / self.res[0])

        if self.render_mode != 'blender' and self.dep_sill_mode != 'blender':
            if rotate:
                ang = (idx / 20) * np.pi * 2
                mv = utils.translate(0, 0, -self.cam_radius) @ (utils.rotate_x(0.4) @ utils.rotate_y(ang))
                mv = mv.repeat(batch, 1, 1)
            else:
                mv = torch.bmm(
                    utils.translate(0, 0, -self.cam_radius).repeat(batch, 1, 1),
                    utils.batch_random_rotation_translation(self.camera_angle_num, 0.25)
                )
            depths = torch.zeros((batch,) + tuple(self.res) + (1,))
            imgs = torch.zeros((batch,) + tuple(self.res) + (3,))
        else:
            mv = self.data[num].get('gt_transforms')
            img_num = len(mv)
            index = random.sample(range(img_num), k=1 if rotate else self.camera_angle_num)

            mv = mv[index]
            imgs = self.data[num].get('gt_render_imgs')[index]
            depths = self.data[num].get('gt_render_depths')[index]

        mvp = torch.bmm(
            proj_mtx.repeat(batch, 1, 1),
            mv
        )
        campos = torch.linalg.inv(mv)[:, :3, 3]

        return mv.to(self.device), mvp.to(self.device), campos.to(self.device), \
               imgs.to(self.device), depths.to(self.device), proj_mtx.to(self.device), self.res

    def __len__(self):
        # return self.epoch if self.validate else self.its_per_epc
        return 20 if self.visual else len(self.data)

    def __getitem__(self, idx):
        if self.visual:
            num = idx % len(self.data)
        else:
            num = idx

        # rotate or random scene
        mv, mvp, campos, imgs, depths, persp, iter_res = self._sample_scene(idx, num, rotate=self.visual)

        if self.dep_sill_mode == 'blender':
            alpha = alpha_ctn = (depths > 0).float()
            normal, face_normals = renderer.render_mesh(
                self.glctx,
                self.data[num]['gt_verts'].to(self.device),
                self.data[num]['gt_faces'].to(self.device),
                mvp, campos, persp, iter_res,
                normal_only=True
            )
        else:
            alpha, alpha_ctn, depths, normal, face_normals = renderer.render_mesh(
                self.glctx,
                self.data[num]['gt_verts'].to(self.device),
                self.data[num]['gt_faces'].to(self.device),
                mvp, campos, persp, iter_res
            )

        # pcd_num = len(self.data[num]['gt_pcd'])
        # index = random.sample(range(pcd_num), k=self.pts_num)
        # index_query = random.sample(range(pcd_num), k=min(self.query_pts_num, self.pts_num))

        res = {
            'uuid': self.data[num].get('uuid'),
            'gt_verts': self.data[num]['gt_verts'],
            'gt_faces': self.data[num]['gt_faces'],
            # 'gt_pcd': self.data[num]['gt_pcd'][None, index, ...],
            # 'gt_pcd_colors': self.data[num]['gt_pcd_colors'][None, index, ...],
            # 'gt_pcd_colors_query': self.data[num]['gt_pcd_colors'][None, index_query, ],
            # 'gt_pcd_query': self.data[num]['gt_pcd'][None, index_query, ],
            'gt_render_imgs': imgs[None, ],
            'mv': mv[None, ],
            'mvp': mvp[None, ],
            'campos': campos[None, ],
            'resolution': self.res,
            'resolution_img': self.res_img,
            'resolution_triview': self.res_triview,
            'normal': normal[None, ],
            'alpha': alpha[None, ],
            'alpha_ctn': alpha_ctn[None, ],
            'depth': depths[None, ],
            'persp': persp
        }

        if self.triview_color_dir != '':
            triview_color = self.data[num]['triview_color']
            res.update({
                'triview_color': triview_color[None, ],
            })

        if self.triview_xyz_dir != '':
            triview_xyz = self.data[num]['triview_xyz']
            res.update({
                'triview_xyz': triview_xyz[None, ],
            })

        if self.prompt_path != '':
            prompt = self.data[num]['prompt']
            res.update({
                'prompt': [prompt]
            })

        return res
