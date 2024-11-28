'''
Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
'''

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
        """custom collation for non torch values like resolution, perspectives"""
        res = {
            "uuid": list(item["uuid"] for item in batch),
            "verts": list(item["verts"].to(self.device) for item in batch),
            "faces": list(item["faces"].to(self.device) for item in batch),
            "render_imgs": torch.cat(list(item["render_imgs"] for item in batch), dim=0).to(self.device),
            "mv": torch.cat(list(item["mv"] for item in batch), dim=0).to(self.device),
            "mvp": torch.cat(list(item["mvp"] for item in batch), dim=0).to(self.device),
            # "campos": torch.cat(list(item["campos"] for item in batch), dim=0).to(self.device),
            "alpha": torch.cat(list(item["alpha"] for item in batch), dim=0).to(self.device),
            # "alpha_ctn": torch.cat(list(item["alpha_ctn"] for item in batch), dim=0).to(self.device),
            "depth": torch.cat(list(item["depth"] for item in batch), dim=0).to(self.device),
            "normal": torch.cat(list(item["normal"] for item in batch), dim=0).to(self.device),
            "triview_color": torch.cat(list(item["triview_color"] for item in batch), dim=0).to(self.device),
            "triview_xyz": torch.cat(list(item["triview_xyz"] for item in batch), dim=0).to(self.device),
            "prompt": list(item["prompt"] for item in batch),
            "persp": batch[0]["persp"].to(self.device),
            "resolution": batch[0]["resolution"],
            "resolution_img": batch[0]["resolution_img"],
            "resolution_triview": batch[0]["resolution_triview"],
        }
        return res
