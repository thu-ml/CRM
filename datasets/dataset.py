# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch


class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, itr):
        raise NotImplementedError

    def collate(self, batch):
        # iter_res = batch[0]['resolution']
        # tets = batch[0]['tets']
        # points = batch[0]['points']
        # iter_spp = batch[0]['spp']
        res = {
            'uuid': list([item['uuid'] for item in batch]),
            # 'gt_pcd': torch.cat(list([item['gt_pcd'] for item in batch]), dim=0).to(self.device),
            # 'gt_pcd_colors':  torch.cat(list([item['gt_pcd_colors'] for item in batch])).to(self.device),
            # 'gt_pcd_query': torch.cat(list([item['gt_pcd_query'] for item in batch]), dim=0).to(self.device),
            # 'gt_pcd_colors_query': torch.cat(list([item['gt_pcd_colors_query'] for item in batch])).to(self.device),
            'gt_render_imgs': torch.cat(list([item['gt_render_imgs'] for item in batch]), dim=0).to(self.device),
            # 'verts': torch.cat(list([item['verts'] for item in batch]), dim=0).to(self.device),
            # 'tets': tets.to(self.device),
            'gt_verts': list([item['gt_verts'].to(self.device) for item in batch]),
            'gt_faces': list([item['gt_faces'].to(self.device) for item in batch]),
            'mv': torch.cat(list([item['mv'] for item in batch]), dim=0).to(self.device),
            'mvp': torch.cat(list([item['mvp'] for item in batch]), dim=0).to(self.device),
            'campos': torch.cat(list([item['campos'] for item in batch]), dim=0).to(self.device),
            'alpha': torch.cat(list([item['alpha'] for item in batch]), dim=0).to(self.device),
            'alpha_ctn': torch.cat(list([item['alpha_ctn'] for item in batch]), dim=0).to(self.device),
            'depth': torch.cat(list([item['depth'] for item in batch]), dim=0).to(self.device),
            'normal': torch.cat(list([item['normal'] for item in batch]), dim=0).to(self.device),
            'persp': batch[0]['persp'].to(self.device),
            # 'time': sum(list([item['time'] for item in batch])),
            'resolution': batch[0]['resolution'],
            'resolution_img': batch[0]['resolution_img'],
            'resolution_triview': batch[0]['resolution_triview']
            # 'spp' : iter_spp,
            # 'img': torch.cat(list([item['img'] for item in batch]), dim=0)
        }
        if batch[0].get('triview_color') is not None:
            res.update({
                'triview_color': torch.cat(list([item['triview_color'] for item in batch]), dim=0).to(self.device)
            })
        if batch[0].get('triview_xyz') is not None:
            res.update({
                'triview_xyz': torch.cat(list([item['triview_xyz'] for item in batch]), dim=0).to(self.device)
            })
        if batch[0].get('prompt') is not None:
            res.update({
                'prompt': list([item['prompt'] for item in batch])
            })
        return res


# class DatasetPoints(torch.utils.data.Dataset):
#     """Basic dataset interface"""
#     def __init__(self):
#         super().__init__()
#
#     def __len__(self):
#         raise NotImplementedError
#
#     def __getitem__(self, itr):
#         raise NotImplementedError
#
#     def collate(self, batch):
#         iter_res = batch[0]['resolution']
#         # iter_spp = batch[0]['spp']
#         return {
#             'mv': torch.cat(list([item['mv'] for item in batch]), dim=0),
#             'mvp': torch.cat(list([item['mvp'] for item in batch]), dim=0),
#             'campos': torch.cat(list([item['campos'] for item in batch]), dim=0),
#             'resolution': iter_res,
#             # 'spp' : iter_spp,
#             # 'img': torch.cat(list([item['img'] for item in batch]), dim=0)
#         }
