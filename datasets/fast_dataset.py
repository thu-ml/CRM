'''
Chunk Generation for Large Dataset
'''

import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from typing import List, Tuple
import random
import pickle
import logging
import asyncio
import io

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq
import torch
import trimesh
from tqdm import tqdm
import aiofiles
import cv2

from util import utils, renderer
from util.utils import DEPTH_MAX, DEPTH_MIN, DEPTH_VALID_MIN, DEPTH_VALID_MAX
from datasets.dataset import Dataset

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def chunk_with_size(lst, n):
    '''return successive n-sized chunks from lst.'''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_with_number(lst, n):
    '''split lst into n chunks, all chunks have near chunk size diffrence no more than 1, just like np.array_split'''
    chunk_size = len(lst) // n
    remain = len(lst) % n
    start = 0
    for i in range(n):
        end = start + chunk_size
        if i < remain:
            end += 1
        yield lst[start:end]
        start = end

async def aread_file(file_path, mode='rb'):
    '''read files from disk into memory'''
    async with aiofiles.open(file_path, mode=mode) as f:
        return await f.read()

async def abatch_read_files(file_paths: List[str]):
    '''read files from disk into memory (batch), return a list of bytes'''
    return await asyncio.gather(*[aread_file(file_path) for file_path in file_paths])

def buffer_to_cv2image(buffer):
    '''convert bytes to image'''
    return cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def batch_buffer_to_cv2image(buffers):
    '''convert bytes to image (batch)'''
    return [buffer_to_cv2image(buffer) for buffer in buffers]


class FastDataset(Dataset):
    '''
    Torch Dataset for training
    '''
    def __init__(self, input_specs, train_specs, glctx, device, chunk, mode, num_process=1, process_index=0):
        super().__init__(device)
        self.device = device
        self.glctx = glctx
        self.mode = mode
        # self.cam_radius = input_specs['radius']
        self.fovx = np.deg2rad(45)
        self.res = input_specs['resolution']
        self.res_img = input_specs.get('resolution_img')
        self.res_triview = input_specs.get('resolution_triview')
        self.render_color_img_dir = input_specs.get('render_img_dir', '')
        self.triview_color_dir = input_specs.get('triview_color_dir', '')
        self.triview_xyz_dir = input_specs.get('triview_xyz_dir', '')
        self.prompt_path = input_specs.get('prompt_path', '')
        self.aspect = self.res[1] / self.res[0]
        self.pts_num = train_specs['sample_points_num']
        # self.query_pts_num = train_specs['query_sample_points_num']
        self.epoch = train_specs['num_epochs']
        self.scale = input_specs['scale']
        # self.render_mode = train_specs.get('tex_sup_mode')
        # self.dep_sill_mode = train_specs.get('dep&sill_sup_mode')

        self.mesh_dir = input_specs['mesh_dir']
        self.class_name = input_specs['class']
        self.camera_angle_num = input_specs['camera_angle_num']
        self.split_dir = input_specs['split_dir']

        self.train_set = utils.load_item(os.path.join(self.split_dir, 'train.txt'))
        self.val_set = utils.load_item(os.path.join(self.split_dir, 'val.txt'))
        self.test_set = utils.load_item(os.path.join(self.split_dir, 'test.txt'))

        random.seed(2023)
        self.val_memory = sorted(list(self.val_set))
        self.validate_num = input_specs.get('validate_num')
        self.val_memory = random.sample(self.val_memory, k=self.validate_num)

        self.objaverse_exclude = utils.load_item(os.path.join(self.split_dir, 'exclude.txt'))
        self.all_set = self.train_set.union(self.val_set).union(self.test_set) - set(self.val_memory)

        self.img_num = input_specs.get('img_num', 24)

        self.uuid2transform = utils.load_transform(input_specs.get('blender_transform'))
        chunk_path = input_specs['chunk_path']
        self.dataset_name = input_specs['dataset_name']

        class_name_suffix = f'_reso{self.res[0]}_img{self.img_num}_fast'
        class_name = input_specs['class'] + class_name_suffix

        self.num_process = num_process
        self.process_index = process_index

        if self.prompt_path != '':
            self.uuid2prompt = utils.load_prompt(self.prompt_path)

        chunk_size = input_specs['chunk_size']
        parquet_dir = os.path.join(chunk_path, self.dataset_name, class_name, self.mode)

        if not os.path.exists(parquet_dir):
            os.makedirs(parquet_dir, exist_ok=True)

        parquet_paths = [os.path.join(parquet_dir, name) for name in os.listdir(parquet_dir) if name.endswith('.parquet')]
        self.num_chunk = len(parquet_paths)
        self._parquet_paths = parquet_paths
        self.metadata = []

        self.parquet_dir = parquet_dir
        self.chunk_size = chunk_size

        self._parquet_paths.sort()

        chunk_indices = list(range(len(self._parquet_paths)))
        self._chunk_index = cycle(chunk_indices[chunk:] + chunk_indices[:chunk])
        self._loaded_uuid = None
        self._loaded_verts_list = None
        self._loaded_faces_list = None
        self._loaded_render_color_img_list = None
        self._loaded_render_depth_img_list = None
        self._loaded_render_color_cam_list = None
        self._loaded_fovx_list = None
        self._loaded_triview_color_img_list = None
        self._loaded_triview_xyz_img_list = None
        self._loaded_prompt = None
        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)
        self._chosen = None

    def load_chunk(self) -> None:
        '''execute chunk generation and prepare for next task'''
        chosen, self._loaded_verts_list, self._loaded_faces_list, \
            self._loaded_render_color_img_list, self._loaded_render_depth_img_list, \
            self._loaded_render_color_cam_list, self._loaded_fovx_list, \
            self._loaded_triview_color_img_list, self._loaded_triview_xyz_img_list, self._loaded_prompt, self._loaded_uuid= \
            self._chunk_future.result()
        self._chosen = chosen
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)

    # def get_state(self) -> str:
    #     return self._chosen

    # def set_state(self, chosen: str) -> None:
    #     while self._chosen != chosen:
    #         self.load_chunk()

    def __len__(self) -> int:
        return len(self._loaded_verts_list)

    def _random_scene(self, mv=None, imgs=None, depths=None):
        #  Setup projection matrix
        proj_mtx = utils.perspective(self.fovx, self.res[1] / self.res[0])

        # Random rotation/translation matrix for optimization.
        batch = self.camera_angle_num
        # if mv is None or imgs is None:
        #     mv = torch.bmm(
        #         utils.translate(0, 0, -self.cam_radius).repeat(batch, 1, 1),
        #         utils.batch_random_rotation_translation(self.camera_angle_num, 0.25)
        #     )
        #     depths = torch.zeros((batch,) + tuple(self.res) + (1,))
        #     imgs = torch.zeros((batch,) + tuple(self.res) + (3,))
        # else:
        img_num = len(mv)
        index = random.sample(range(img_num), k=self.camera_angle_num)

        mv = mv[index]
        imgs = imgs[index]
        depths = depths[index]

        mvp = torch.bmm(
            proj_mtx.repeat(batch, 1, 1),
            mv
        )
        # campos = torch.linalg.inv(mv)[:, :3, 3]

        return mv.to(self.device), mvp.to(self.device), imgs.to(self.device), \
            depths.to(self.device), proj_mtx.to(self.device), self.res

    def __getitem__(self, idx):
        # if self.render_mode == 'blender':
        mv = self._loaded_render_color_cam_list[idx]
        imgs = self._loaded_render_color_img_list[idx]
        depths = self._loaded_render_depth_img_list[idx]
        triview_color = self._loaded_triview_color_img_list[idx]
        triview_xyz = self._loaded_triview_xyz_img_list[idx]

        mv = mv.T.view(-1, 4, 4)
        raws = imgs.T.view(imgs.shape[-1], self.res_img[0], self.res_img[1], 4)

        imgs = raws[..., :3]
        masks = raws[..., [-1]].expand(-1, -1, -1, 3)
        imgs *= masks

        depths = depths.T.view(depths.shape[-1], self.res_img[0], self.res_img[1], 1)

        self.fovx = self._loaded_fovx_list[idx][0]

        mv, mvp, imgs, depths, persp, iter_res = self._random_scene(mv, imgs, depths)
        # else:
        #     mv, mvp, campos, imgs, depths, persp, iter_res = self._random_scene()

        verts = self._loaded_verts_list[idx]

        # if self.dataset_name == 'Objaverse':
            # if self.uuid2transform:
        transform = self.uuid2transform[self._loaded_uuid[idx]]
        transport = torch.tensor(transform['offset'], dtype=torch.float32)
        scale = torch.tensor(transform['scale'], dtype=torch.float32)
        verts = (verts * scale + transport) * self.scale / 0.5
            # else:
            #     transport = (verts.max(0)[0] + verts.min(0)[0]) / 2
            #     verts -= transport
            #     scale = 1 / max(verts.max(0)[0] - verts.min(0)[0])

            #     verts *= self.scale / 0.5 * scale

        # if self.dep_sill_mode == 'blender':
        #     alpha = alpha_ctn = (depths > 0).float()
        #     normal, face_normals = renderer.render_mesh(
        #         self.glctx, verts.to(self.device),
        #         self._loaded_faces_list[idx].to(self.device),
        #         mvp, campos, persp, iter_res,
        #         normal_only=True
        #     )
        # else:
        alpha, depth, normal = renderer.render_mesh(
            self.glctx, verts.to(self.device),
            self._loaded_faces_list[idx].to(self.device),
            mvp, persp, iter_res
        )

        rgbas = triview_color.T.view(triview_color.shape[-1], self.res_triview[0], self.res_triview[1], 4)
        triview_color = rgbas[..., :3]
        masks = rgbas[..., [-1]].expand(-1, -1, -1, 3)
        triview_color *= masks

        triview_xyz = triview_xyz.T.view(triview_xyz.shape[-1], self.res_triview[0], self.res_triview[1], 3)
        triview_xyz *= masks

        res = {
            'uuid': self._loaded_uuid[idx],
            'verts': self._loaded_verts_list[idx],
            'faces': self._loaded_faces_list[idx],
            'render_imgs': imgs[None,],
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
            'triview_color': triview_color[None,],
            'triview_xyz': triview_xyz[None,],
            'prompt': [self.uuid2prompt[self._loaded_uuid[idx]]]
        }
        return res


    def _load_chunk_inner(self) -> Tuple[str, List[torch.FloatTensor], List[torch.LongTensor],
                                         List[torch.FloatTensor], List[torch.FloatTensor],
                                         List[torch.FloatTensor], List[torch.FloatTensor],
                                         List[torch.FloatTensor], List[torch.FloatTensor],
                                         List[torch.FloatTensor], List[torch.FloatTensor], List[str], List[str]]:

        next_index = next(self._chunk_index)
        chosen = self._parquet_paths[next_index]

        loaded_chunk = pq.read_table(chosen)
        num_rows = loaded_chunk.num_rows
        pandas_chunk = loaded_chunk.to_pandas()

        verts_list = [torch.FloatTensor(np.stack(
            [pandas_chunk[f'verts_list_{j}'][i] for j in range(3)], axis=-1)
        ) for i in range(num_rows)]
        faces_list = [torch.LongTensor(np.stack(
            [pandas_chunk[f'faces_list_{j}'][i] for j in range(3)], axis=-1)
        ) for i in range(num_rows)]
        uuid = [str(pandas_chunk['uuid'][i]) for i in range(num_rows)]

        render_color_img_list, render_depth_img_list, render_color_cam_list, fovx_list = None, None, None, None
        # if self.render_mode == 'blender':
        render_color_img_list = [torch.FloatTensor(np.stack(
            [pandas_chunk[f'render_color_img_list_{j}'][i] for j in range(self.img_num)], axis=-1)
        ) for i in range(num_rows)]

        render_depth_img_list = [torch.FloatTensor(np.stack(
            [pandas_chunk[f'render_depth_img_list_{j}'][i] for j in range(self.img_num)], axis=-1)
        ) for i in range(num_rows)]

        render_color_cam_list = [torch.FloatTensor(np.stack(
            [pandas_chunk[f'render_color_cam_list_{j}'][i] for j in range(self.img_num)], axis=-1)
        ) for i in range(num_rows)]

        fovx_list = [torch.FloatTensor(np.array(pandas_chunk['fovx_list'][i])) for i in range(num_rows)]

        triview_color_img_list, triview_xyz_img_list, prompt = None, None, None
        # if self.triview_color_dir != '':
        triview_color_img_list = [torch.FloatTensor(np.stack(
            [pandas_chunk[f'triview_color_img_list_{j}'][i] for j in range(6)], axis=-1)
        ) for i in range(num_rows)]

        # if self.triview_xyz_dir != '':
        triview_xyz_img_list = [torch.FloatTensor(np.stack(
            [pandas_chunk[f'triview_xyz_img_list_{j}'][i] for j in range(6)], axis=-1)
        ) for i in range(num_rows)]

        # if self.prompt_path != '':
        prompt = [str(pandas_chunk['prompt'][i]) for i in range(num_rows)]

        return str(chosen), verts_list, faces_list, \
               render_color_img_list, render_depth_img_list, render_color_cam_list, fovx_list, \
               triview_color_img_list, triview_xyz_img_list, prompt, uuid

    async def aprocess_one_data(self, data: dict, data_index:int):
        '''
        given data, read files and preprocess them, then put them into lists
        '''
        if data_index % 100 == 0:
            logging.info('process index %d start %d', self.process_index, data_index)

        gt_path = data['gt_path']

        gt_bytes = await aread_file(gt_path)
        ref_mesh = trimesh.load(io.BytesIO(gt_bytes),
                                file_type=Path(gt_path).suffix, force='mesh', process=False)
        verts = torch.tensor(ref_mesh.vertices, dtype=torch.float32)
        faces = torch.LongTensor(ref_mesh.faces)

        render_imgs = None
        render_depths = None
        transforms = None
        fovx = None
        if self.render_color_img_dir != '':
            render_color_img_dir = data['render_color_img_dir']
            render_color_cam_file = data['render_color_cam_file']
            # rotate
            verts[:, -1] = -verts[:, -1]
            verts = verts[:, [0, 2, 1]]

            # if self.dataset_name == 'Objaverse':
            k, _, _, _, w2cs = pickle.load(io.BytesIO(await aread_file(render_color_cam_file)))

            fovx = np.array([2 * math.atan(k[0, 2] / k[0, 0])], dtype=np.float64)
            img_list, depth_list, transform_list = [], [], []
            image_paths = [os.path.join(render_color_img_dir, f'{i:03d}.png') for i in range(len(w2cs))]
            bytes_images = await abatch_read_files(image_paths)
            images = batch_buffer_to_cv2image(bytes_images)

            depth_paths = [os.path.join(render_color_img_dir, f'{i:03d}-depth.png') for i in range(len(w2cs))]
            bytes_depths = await abatch_read_files(depth_paths)
            depths = batch_buffer_to_cv2image(bytes_depths)

            # select per stride imgs
            assert len(w2cs) % self.img_num == 0
            stride = len(w2cs) // self.img_num

            for i, w2c in enumerate(w2cs):
                if i % stride != 0:
                    continue
                w2c = torch.tensor(np.stack((w2c[0], -w2c[1], -w2c[2], np.array([0., 0., 0., 1.])), axis=0),
                                    dtype=torch.float32)
                c2w = torch.inverse(w2c)
                c2w[:3, 3] *= self.scale / 0.5
                w2c = torch.inverse(c2w)

                raw = cv2.resize(images[i], tuple(self.res_img))
                img = torch.tensor(raw[:, :, [2, 1, 0, 3]],
                                    dtype=torch.float32) / 255

                transform_list.append(w2c.reshape(-1))
                img_list.append(img.reshape(-1))

                # depth
                # syncdreamer
                # if os.path.exists(os.path.join(render_color_img_dir, f'{i:03d}-depth.png')):
                # depth_name = f'{i:03d}-depth.png'
                # dep = cv2.imread(os.path.join(render_color_img_dir, depth_name), cv2.IMREAD_UNCHANGED)
                dep = depths[i]

                # reference
                # https://github.com/liuyuan-pal/SyncDreamer/blob/232a2ec22f0277b4bd429109fb34eaf8f61b13ad/eval_mesh.py#L16-L22
                dep = dep.astype(np.float32) / 65535 * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
                mask = (dep < DEPTH_VALID_MIN) | (dep > DEPTH_VALID_MAX)

                # # cs
                # elif os.path.exists(os.path.join(render_color_img_dir, f'depth_{i:03d}1.png')):
                #     depth_name = f'{i:03d}-depth.png'
                #     depth_path = os.path.join(render_color_img_dir, f'depth_{i:03d}1.png')

                #     dep =(np.array(Image.open(depth_path))[:,:,0]- 120)* 0.5 / 63
                #     mask = np.array(Image.open(depth_path))[:,:,0] == 255

                dep = np.where(mask, 0., dep)

                dep = torch.tensor(cv2.resize(dep, tuple(self.res)), dtype=torch.float32) * self.scale / 0.5

                depth_list.append(dep.reshape(-1))
            render_imgs = torch.stack(img_list, dim=-1)
            transforms = torch.stack(transform_list, dim=-1)
            render_depths = torch.stack(depth_list, dim=-1)

        triview_imgs = None
        triview_xyzs = None
        # if self.triview_color_dir != '' and self.triview_xyz_dir != '':
        img_list, xyz_list = [], []
        triview_color_dir = data['triview_color_dir']
        triview_xyz_dir = data['triview_xyz_dir']

        color_paths = [os.path.join(triview_color_dir, f'{i:03d}.png') for i in range(6)]
        xyz_paths = [os.path.join(triview_xyz_dir, f'xyz_new_{i:03d}.png') for i in range(6)]

        for i in range(6):
            # BGR -> RGB
            color = buffer_to_cv2image(await aread_file(color_paths[i]))
            xyz = buffer_to_cv2image(await aread_file(xyz_paths[i]))
            # raw = cv2.resize(colors[i], self.res_triview)
            # xyz = cv2.resize(xyzs[i], self.res_triview)
            raw = cv2.resize(color, self.res_triview)
            xyz = cv2.resize(xyz, self.res_triview)

            # mask = torch.tensor(raw[:, :, [-1]],
                                # dtype=torch.float32).repeat(1, 1, 3) / 255
            img = torch.tensor(raw[:, :, [2, 1, 0, 3]],
                                dtype=torch.float32) / 255
            xyz = torch.tensor(xyz, dtype=torch.float32) / 255

            # img = img * mask
            # xyz = xyz * mask

            img_list.append(img.reshape(-1))
            xyz_list.append(xyz.reshape(-1))

        triview_imgs = torch.stack(img_list, dim=-1)
        triview_xyzs = torch.stack(xyz_list, dim=-1)

        verts = verts.numpy().astype(np.float32)
        faces = faces.numpy().astype(np.int64)

        render_imgs = render_imgs.numpy().astype(np.float32) if render_imgs is not None else None
        render_depths = render_depths.numpy().astype(np.float32) if render_depths is not None else None
        transforms = transforms.numpy().astype(np.float32) if transforms is not None else None

        triview_imgs = triview_imgs.numpy().astype(np.float32) if triview_imgs is not None else None
        triview_xyzs = triview_xyzs.numpy().astype(np.float32) if triview_xyzs is not None else None

        if data_index % 100 == 0:
            logging.info('process index %d start %d', self.process_index, data_index)

        return {
            'data_index': data_index,
            'uuid': data['uuid'],
            'verts': verts,
            'faces': faces,
            'fovx': fovx,
            'transforms': transforms,
            'render_imgs': render_imgs,
            'render_depths': render_depths,
            'triview_imgs': triview_imgs,
            'triview_xyzs': triview_xyzs,
            'prompt': data.get('prompt')
        }

    def append_data(self, data: dict,
                    uuid, verts_list, faces_list,
                    render_color_img_list, render_depth_img_list, render_color_cam_list,
                    fovx_list, triview_color_img_list, triview_xyz_img_list, prompt) -> None:
        '''Collect data'''
        uuid.append(data['uuid'])
        verts_list.append(data['verts'])
        faces_list.append(data['faces'])

        # if self.render_color_img_dir != '':
        render_color_img_list.append(data['render_imgs'])
        render_depth_img_list.append(data['render_depths'])
        render_color_cam_list.append(data['transforms'])
        fovx_list.append(data['fovx'])
        # if self.triview_color_dir != '':
        triview_color_img_list.append(data['triview_imgs'])
        # if self.triview_xyz_dir != '':
        triview_xyz_img_list.append(data['triview_xyzs'])

        # if self.prompt_path != '':
        prompt.append(data['prompt'])

    def prepare_metadata(self):
        '''Prepare all string like metadata e.g. image path, uuid, prompt, etc'''
        for class_name in os.listdir(self.mesh_dir):
            if self.class_name in [class_name, 'all']:
                class_dir = os.path.join(self.mesh_dir, class_name)
                if class_name.endswith('.json'):
                    continue
                for object_name in tqdm(os.listdir(class_dir)):
                    # if i > 999:
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

                    if (self.mode == 'train' and record in self.train_set) or \
                            (self.mode == 'val' and record in self.val_set) or \
                            (self.mode == 'test' and record in self.test_set) or \
                            (self.mode == 'all' and record in self.all_set):

                        # load vertices and faces
                        gt_path = os.path.join(class_dir, object_name)
                        # pcd_file = os.path.join(self.pcd_dir, record, 'pc_262144_131072.ply')

                        render_color_img_dir = os.path.join(self.render_color_img_dir, record)
                        render_color_cam_file = os.path.join(self.render_color_img_dir, record, 'meta.pkl')

                        triview_color_dir = os.path.join(self.triview_color_dir, record)

                        triview_xyz_dir = os.path.join(self.triview_xyz_dir, record)

                        prompt = self.uuid2prompt.get(record, None)
                        res = {
                            'gt_path': gt_path,
                            'uuid': record,
                            'render_color_img_dir': render_color_img_dir,
                            'render_color_cam_file': render_color_cam_file,
                            'triview_color_dir': triview_color_dir,
                            'triview_xyz_dir': triview_xyz_dir,
                            'prompt': prompt
                        }
                        self.metadata.append(res)
        self.metadata.sort(key=lambda x: x['gt_path'])

    def write_one_chunk(self, chunk_index:int,
                        verts_list: List[torch.FloatTensor], faces_list: List[torch.LongTensor],
                        render_color_img_list: List[torch.FloatTensor], render_depth_img_list: List[torch.FloatTensor],
                        render_color_cam_list: List[torch.FloatTensor],
                        fovx_list: List[torch.DoubleTensor], triview_color_img_list: List[torch.FloatTensor],
                        triview_xyz_img_list: List[torch.FloatTensor], uuid: List[str], prompt: List[str],
                        parquet_writers: List[pq.ParquetWriter]):
        '''Gather one chunk data to generate a pyarrow table and save to the disk'''
        logging.info('----process %d start write index %d %s----',
                     self.process_index, chunk_index, parquet_writers[chunk_index].where)
        indices = torch.randperm(len(verts_list)).tolist()
        shuffled_verts_list = [verts_list[i] for i in indices]
        shuffled_faces_list = [faces_list[i] for i in indices]
        shuffled_uuid = [uuid[i] for i in indices]
        # if self.render_color_img_dir != '':
        shuffled_render_color_img_list = [render_color_img_list[i] for i in indices]
        shuffled_render_depth_img_list = [render_depth_img_list[i] for i in indices]
        shuffled_render_color_cam_list = [render_color_cam_list[i] for i in indices]
        shuffled_fovx_list = [fovx_list[i] for i in indices]
        # if self.triview_color_dir != '':
        shuffled_triview_color_img_list = [triview_color_img_list[i] for i in indices]
        # if self.triview_xyz_dir != '':
        shuffled_triview_xyz_img_list = [triview_xyz_img_list[i] for i in indices]
        # if self.prompt_path != '':
        shuffled_prompt = [prompt[i] for i in indices]

        columns = {}
        columns['uuid'] = shuffled_uuid
        for i in range(3):
            columns[f'verts_list_{i}'] = [verts[:, i] for verts in shuffled_verts_list]
            columns[f'faces_list_{i}'] = [faces[:, i] for faces in shuffled_faces_list]

        # if self.render_color_img_dir != '':
        for i in range(self.img_num):
            columns[f'render_color_img_list_{i}'] = [
                render_color_img[:, i] for render_color_img in
                shuffled_render_color_img_list
            ]
            columns[f'render_depth_img_list_{i}'] = [
                render_depth_img[:, i] for render_depth_img in
                shuffled_render_depth_img_list
            ]
            columns[f'render_color_cam_list_{i}'] = [
                render_color_cam[:, i] for render_color_cam in
                shuffled_render_color_cam_list
            ]
        columns['fovx_list'] = shuffled_fovx_list

        # if self.triview_color_dir != '':
        for i in range(6):
            columns[f'triview_color_img_list_{i}'.format(i)] = [
                triview_color_img[:, i] for triview_color_img in
                shuffled_triview_color_img_list
            ]
        # if self.triview_xyz_dir != '':
        for i in range(6):
            columns[f'triview_xyz_img_list_{i}'.format(i)] = [
                triview_xyz_img[:, i] for triview_xyz_img in
                shuffled_triview_xyz_img_list
            ]
        # if self.prompt_path != '':
        columns['prompt'] = shuffled_prompt
        parquet_writers[chunk_index].write_table(pa.table(columns))
        parquet_writers[chunk_index].close()
        logging.info('____process %d start write index %d %s____',
                     self.process_index, chunk_index, parquet_writers[chunk_index].where)

    def write_chunks(self, chunk_path, chunk_size) -> None:
        '''Chunk generation and save main function'''
        logging.info('start')
        self.prepare_metadata()
        logging.info('first for done')
        assert ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        total_chunks = list(chunk_with_size(self.metadata, chunk_size))
        chunks_for_process = list(chunk_with_number(total_chunks, self.num_process))
        current_process_chunks = chunks_for_process[self.process_index]

        parquet_writers = []

        logging.info('process_index: %d, Allocating %d chunks, with totol size of %d to dataset path %s', self.process_index,
                     len(current_process_chunks), sum([len(chunk) for chunk in current_process_chunks]), chunk_path)

        j = 0
        for p_index in range(self.num_process):
            for chunk in chunks_for_process[p_index]:
                parquet_path = os.path.join(chunk_path, f'{j:06d}.parquet')
                j += 1
                if p_index != self.process_index:
                    continue
                self._parquet_paths.append(parquet_path)
                dtypes = []
                dtypes.append(('uuid', pa.string()))
                for i in range(3):
                    dtypes.append((f'verts_list_{i}', pa.list_(pa.from_numpy_dtype(np.float32))))
                    dtypes.append((f'faces_list_{i}', pa.list_(pa.from_numpy_dtype(np.int64))))
                # if self.render_color_img_dir != '':
                for i in range(self.img_num):
                    dtypes.append((f'render_color_img_list_{i}', pa.list_(pa.from_numpy_dtype(np.float32))))
                    dtypes.append((f'render_depth_img_list_{i}', pa.list_(pa.from_numpy_dtype(np.float32))))
                    dtypes.append((f'render_color_cam_list_{i}', pa.list_(pa.from_numpy_dtype(np.float32))))
                dtypes.append(('fovx_list', pa.list_(pa.from_numpy_dtype(np.float64))))
                # if self.triview_color_dir != '':
                for i in range(6):
                    dtypes.append((f'triview_color_img_list_{i}', pa.list_(pa.from_numpy_dtype(np.float32))))
                # if self.triview_xyz_dir != '':
                for i in range(6):
                    dtypes.append((f'triview_xyz_img_list_{i}', pa.list_(pa.from_numpy_dtype(np.float32))))
                # if self.prompt_path != '':
                dtypes.append(('prompt', pa.string()))
                parquet_writers.append(pq.ParquetWriter(parquet_path, pa.schema(dtypes), compression='BROTLI'))

        bar_write = tqdm(total=len(current_process_chunks))
        bar_write.set_description('write')

        def do_one_chunk(obj, chunk, chunk_index):
            nonlocal bar_write
            verts_list, faces_list = [], []
            render_color_img_list, render_depth_img_list, render_color_cam_list, fovx_list = [], [], [], []
            triview_color_img_list, triview_xyz_img_list = [], []
            uuid, prompt = [], []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            futures = [obj.aprocess_one_data(chunk[i], i) for i in range(len(chunk))]
            readed_data = loop.run_until_complete(asyncio.gather(*futures))
            loop.close()
            for data in readed_data:
                obj.append_data(data, uuid, verts_list, faces_list,
                            render_color_img_list, render_depth_img_list, render_color_cam_list, fovx_list,
                            triview_color_img_list, triview_xyz_img_list, prompt)

            obj.write_one_chunk(chunk_index,
                                verts_list, faces_list,
                                render_color_img_list, render_depth_img_list, render_color_cam_list, fovx_list,
                                triview_color_img_list, triview_xyz_img_list, uuid, prompt,
                                parquet_writers)
            bar_write.update(1)

        task_futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for chunk_index, chunk in enumerate(tqdm(current_process_chunks)):
                # For debug use
                # do_one_chunk(self, chunk, chunk_index)
                task_futures.append(executor.submit(do_one_chunk, self, chunk, chunk_index))

            # wait for all tasks done
            for task in task_futures:
                task.result()

        print('Finished writing chunks to dataset paths')
