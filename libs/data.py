import numpy
import numpy as np
import torch
import os
import random
import pandas as pd
import os.path as osp
import PIL.Image as Image
from torch.utils.data import Dataset
from pathlib import Path
from imagedream.ldm.util import add_random_background
from imagedream.camera_utils import get_camera_for_index
from libs.base_utils import do_resize_content, add_stroke

import torchvision.transforms as transforms


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = numpy.random.randint(
            127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8
        )
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def axis_rotate_xyz(img: Image.Image, rotate_axis="z", angle=90.0):
    img = img.convert("RGB")
    img = np.array(img) - 127
    img = img.astype(np.float32)
    # perform element-wise sin-cos rotation
    if rotate_axis == "z":
        img = np.stack(
            [
                img[..., 0] * np.cos(angle) - img[..., 1] * np.sin(angle),
                img[..., 0] * np.sin(angle) + img[..., 1] * np.cos(angle),
                img[..., 2],
            ],
            -1,
        )
    elif rotate_axis == "y":
        img = np.stack(
            [
                img[..., 0] * np.cos(angle) + img[..., 2] * np.sin(angle),
                img[..., 1],
                -img[..., 0] * np.sin(angle) + img[..., 2] * np.cos(angle),
            ],
            -1,
        )
    elif rotate_axis == "x":
        img = np.stack(
            [
                img[..., 0],
                img[..., 1] * np.cos(angle) - img[..., 2] * np.sin(angle),
                img[..., 1] * np.sin(angle) + img[..., 2] * np.cos(angle),
            ],
            -1,
        )

    return Image.fromarray(img.astype(np.uint8) + 127)


class DataHQCRelative(Dataset):
    """
    - base_dir
        - uid1
            - 000.png
            - 001.png
            - ...
        - uid2
    - xyz_base
        - uid1
            - xyz_new_000.png
            - xyz_new_001.png
            - ...
    accepte caption data(in csv format)
    """

    def __init__(
        self,
        base_dir,
        caption_csv,
        ref_indexs=[0],
        ref_position=-1,
        xyz_base=None,
        camera_views=[3, 6, 9, 12, 15],  # camera views are relative views, not abs
        split="train",
        image_size=256,
        random_background=False,
        resize_rate=1,
        num_frames=5,
        repeat=100,
        outer_file=None,
        debug=False,
        eval_size=100,
    ):
        print(__class__)
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        df = pd.read_csv(caption_csv, sep=",", names=["id", "caption"])
        id_to_caption = {}
        for i in range(len(df.index)):
            item = df.iloc[i]
            id_to_caption[item["id"]] = item["caption"]

        # outer file is txt file, containing each ident per line, specific idents not included in the train process
        outer_set = (
            set(open(outer_file, "r").read().strip().split("\n"))
            if outer_file is not None
            else set()
        )
        xyz_set = set(os.listdir(xyz_base)) if xyz_base is not None else set()
        common_keys = set(id_to_caption.keys()) & set(os.listdir(base_dir))
        common_keys = common_keys & xyz_set if xyz_base is not None else common_keys
        common_keys = common_keys - outer_set
        self.common_keys = common_keys
        self.id_to_caption = id_to_caption
        final_dict = {key: id_to_caption[key] for key in common_keys}
        self.image_size = image_size
        self.base_dir = Path(base_dir)
        self.xyz_base = xyz_base
        self.repeat = repeat
        self.num_frames = num_frames
        self.camera_views = camera_views[:num_frames]
        self.split = split
        self.ref_indexs = ref_indexs
        self.ref_position = ref_position
        self.resize_rate = resize_rate
        self.random_background = random_background
        self.debug = debug
        assert split in ["train", "eval"]

        clip_size = 224
        self.transfrom_clip = transforms.Compose(
            [
                transforms.Resize(
                    (clip_size, clip_size),
                    interpolation=Image.BICUBIC,
                    antialias="warn",
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ]
        )

        self.transfrom_vae = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # 对于第i个视角作为参考时，左边，下面，背面，右边，上面，的图片名称index
        import torchvision.transforms.functional as TF
        from functools import partial as PA

        self.index_mapping = [
            #    正  左 下 背  右 上
            [0, 1, 2, 3, 4, 5],  # 0
            [1, 3, 2, 4, 0, 5],  # 1
            [2, 1, 3, 5, 4, 0],  # 2
            [3, 4, 2, 0, 1, 5],  # 3
            [4, 0, 2, 1, 3, 5],  # 4
            [5, 1, 0, 2, 4, 3],  # 5
        ]
        TT = {
            "r90": PA(TF.rotate, angle=-90.0),  # 顺时针90
            "r180": PA(TF.rotate, angle=-180.0),  # 顺时针180
            "r270": PA(TF.rotate, angle=-270.0),  # 顺时针270
            "s90": PA(TF.rotate, angle=90.0),  # 逆时针90
            "s180": PA(TF.rotate, angle=180.0),  # 逆时针180
            "s270": PA(TF.rotate, angle=270.0),  # 逆时针270
        }

        self.transfroms_mapping = [
            # 正 左 下 背  右 上
            [None, None, None, None, None, None],  # 0
            [None, None, TT["r90"], None, None, TT["s90"]],  # 1
            [None, TT["s90"], TT["s180"], TT["r180"], TT["r90"], None],  # 2
            [None, None, TT["r180"], None, None, TT["s180"]],  # 3
            [None, None, TT["s90"], None, None, TT["r90"]],  # 4
            [None, TT["r90"], None, TT["r180"], TT["s90"], TT["s180"]],  # 5
        ]

        XT = {  # xyz transforms
            "zRrota90": PA(axis_rotate_xyz, rotate_axis="z", angle=np.pi / 2),
            "zSrota90": PA(axis_rotate_xyz, rotate_axis="z", angle=-np.pi / 2),
            "zrota180": PA(axis_rotate_xyz, rotate_axis="z", angle=-np.pi),
            "xRrota90": PA(axis_rotate_xyz, rotate_axis="x", angle=np.pi / 2),
            "xSrota90": PA(axis_rotate_xyz, rotate_axis="x", angle=-np.pi / 2),
        }
        
        self.xyz_transforms_mapping = [
            # 正 左 下 背  右 上
            [None,] * 6, # 0
            [XT["zRrota90"],] * 6, # 1
            [XT["xRrota90"],] * 6, # 2
            [XT["zrota180"],] * 6, # 3
            [XT["zSrota90"],] * 6, # 4
            [XT["xSrota90"],] * 6, # 5
        ]
        
        total_items = [
            {
                "path": os.path.join(base_dir, k),
                "xyz_path": os.path.join(xyz_base, k) if xyz_base is not None else None,
                "caption": v,
            }
            for k, v in final_dict.items()
        ]
        total_items.sort(key=lambda x: x["path"])

        if len(total_items) > eval_size:
            if split == "train":
                self.items = total_items[eval_size:]
            else:
                self.items = total_items[:eval_size]
        else:
            self.items = total_items

        print("============= length of dataset %d =============" % len(self.items))

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, index):
        """
        choose index for target 6 images
        select one of them as input image
        target_images_vae: batch of `num_frame` images of one object from different views, processed by vae_processor
        ref_ip: ref image in piexl space
        ref_ip_img:
        camera views decide the logical camera pose of images:
                000 is front , ev: 0, azimuth: 0
                001 is left  , ev: 0, azimuth: -90
                002 is down  , ev: -90, azimuth: 0
                003 is back  , ev: 0, azimuth: 180
                004 is right , ev: 0, azimuth: 90
                005 is top   , ev: 90, azimuth: 0
        ref_index decides which image choose to be input image

        for example when camera views = [1,   2,   3,   4,  5,  0], ref_position=5
        then dataset return the instance images in order as [left, down, back, right, top, front]
        in which view[ref_position] = view[5] = 0, so the refrence image is the front image

        as all the faces can be rotated to the front face, so any image can be placed to ref_position as ref image(need some tramsforms)
        to do a better control of which image can be placed to ref_position, we can set ref_indexs.
        ref_indexs set [0] default, that means only 000 named images will be placed to ref_position.
        on the situation of ref_indexs=[0, 1, 3, 4], only 000, 001, 003, 004 named images will be placed to ref_position.
        """
        index_mapping = self.index_mapping
        transfroms_mapping = self.transfroms_mapping
        index = index % len(self.items)

        target_dir = self.items[index]["path"]
        target_xyz_dir = self.items[index]["xyz_path"]
        caption = self.items[index]["caption"]

        bg_color = np.random.rand() * 255
        target_images = []
        target_xyz_images = []
        raw_images = []
        raw_xyz_images = []
        alpha_masks = []
        ref_index = random.choice(self.ref_indexs)
        cur_index_mapping = index_mapping[ref_index]
        cur_transfroms_mapping = transfroms_mapping[ref_index]
        cur_xyz_transfroms_mapping = self.xyz_transforms_mapping[ref_index]
        for relative_view in self.camera_views:
            image_index = cur_index_mapping[relative_view]
            trans = cur_transfroms_mapping[relative_view]
            trans_xyz = cur_xyz_transfroms_mapping[relative_view]
            # open
            img = Image.open(
                os.path.join(target_dir, f"{image_index:03d}.png")
            ).convert("RGBA")
            if trans is not None:
                img = trans(img)
            img = do_resize_content(img, self.resize_rate)
            alpha_mask = img.getchannel("A")
            alpha_masks.append(alpha_mask)
            if self.random_background:
                img = add_random_background(img, bg_color)
            img = img.convert("RGB")
            target_images.append(self.transfrom_vae(img))

            raw_images.append(img)

            if self.xyz_base is not None:
                img_xyz = Image.open(
                    os.path.join(target_xyz_dir, f"xyz_new_{image_index:03d}.png")
                ).convert("RGBA")
                img_xyz = trans_xyz(img_xyz) if trans_xyz is not None else img_xyz
                img_xyz = trans(img_xyz) if trans is not None else img_xyz
                img_xyz = do_resize_content(img_xyz, self.resize_rate)
                img_xyz.putalpha(alpha_mask)
                if self.random_background:
                    img_xyz = add_random_background(img_xyz, bg_color)
                img_xyz = img_xyz.convert("RGB")
                target_xyz_images.append(self.transfrom_vae(img_xyz))
                if self.debug:
                    raw_xyz_images.append(img_xyz)

        cameras = [get_camera_for_index(i).squeeze() for i in self.camera_views]
        if self.ref_position is not None:
            cameras[self.ref_position] = torch.zeros_like(
                cameras[self.ref_position]
            )  # set ref camera to zero

        cameras = torch.stack(cameras)

        input_img = Image.open(
            os.path.join(target_dir, f"{ref_index:03d}.png")
        ).convert("RGBA")
        input_img = do_resize_content(input_img, self.resize_rate)
        if self.random_background:
            input_img = add_random_background(input_img, bg_color)
        input_img = input_img.convert("RGB")

        clip_cond = self.transfrom_clip(input_img)
        vae_cond = self.transfrom_vae(input_img)

        vae_target = torch.stack(target_images, dim=0)
        if self.xyz_base is not None:
            xyz_vae_target = torch.stack(target_xyz_images, dim=0)
        else:
            xyz_vae_target = []

        if self.debug:
            print(f"debug!!,{bg_color}")
            return {
                "target_images": raw_images,
                "target_images_xyz": raw_xyz_images,
                "input_img": input_img,
                "cameras": cameras,
                "caption": caption,
                "item": self.items[index],
                "alpha_masks": alpha_masks,
            }

        if self.split == "train":
            return {
                "target_images_vae": vae_target,
                "target_images_xyz_vae": xyz_vae_target,
                "clip_cond": clip_cond,
                "vae_cond": vae_cond,
                "cameras": cameras,
                "caption": caption,
            }
        else:  # eval
            path = os.path.join(target_dir, f"{ref_index:03d}.png")
            return dict(
                path=path,
                target_dir=target_dir,
                cond_raw_images=raw_images,
                cond=input_img,
                ref_index=ref_index,
                ident=f"{index}-{Path(target_dir).stem}",
            )


class DataRelativeStroke(DataHQCRelative):
    """a temp dataset for add sync base using fov data as ref image"""

    def __init__(
        self,
        base_dir,
        caption_csv,
        ref_indexs=[0],
        ref_position=-1,
        xyz_base=None,
        camera_views=[3, 6, 9, 12, 15],  # camera views are relative views, not abs
        split="train",
        image_size=256,
        random_background=False,
        resize_rate=1,
        num_frames=5,
        repeat=100,
        outer_file=None,
        debug=False,
        eval_size=100,
        stroke_p=0.3,
        resize_range=None,
    ):
        print(__class__)
        super().__init__(
            base_dir,
            caption_csv,
            ref_indexs=ref_indexs,
            ref_position=ref_position,
            xyz_base=xyz_base,
            camera_views=camera_views,
            split=split,
            image_size=image_size,
            random_background=random_background,
            resize_rate=resize_rate,
            num_frames=num_frames,
            repeat=repeat,
            outer_file=outer_file,
            debug=debug,
            eval_size=eval_size,
        )
        self.stroke_p = stroke_p
        assert (
            resize_range is None or len(resize_range) == 2
        ), "resize_range should be a tuple of 2 elements"
        self.resize_range = resize_range

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, index):
        index_mapping = self.index_mapping
        transfroms_mapping = self.transfroms_mapping
        index = index % len(self.items)

        target_dir = self.items[index]["path"]
        target_xyz_dir = self.items[index]["xyz_path"]
        caption = self.items[index]["caption"]

        bg_color = np.random.rand() * 255
        target_images = []
        target_xyz_images = []
        raw_images = []
        raw_xyz_images = []
        alpha_masks = []
        ref_index = random.choice(self.ref_indexs)
        cur_index_mapping = index_mapping[ref_index]
        cur_transfroms_mapping = transfroms_mapping[ref_index]
        cur_xyz_transfroms_mapping = self.xyz_transforms_mapping[ref_index]
        cur_resize_rate = (
            random.uniform(*self.resize_range) * self.resize_rate
            if self.resize_range is not None
            else self.resize_rate
        )
        for relative_view in self.camera_views:
            image_index = cur_index_mapping[relative_view]
            trans = cur_transfroms_mapping[relative_view]
            trans_xyz = cur_xyz_transfroms_mapping[relative_view]
            # open
            img = Image.open(
                os.path.join(target_dir, f"{image_index:03d}.png")
            ).convert("RGBA")
            if trans is not None:
                img = trans(img)
            img = do_resize_content(img, cur_resize_rate)
            alpha_mask = img.getchannel("A")
            alpha_masks.append(alpha_mask)
            if self.random_background:
                img = add_random_background(img, bg_color)

            img = img.convert("RGB")
            target_images.append(self.transfrom_vae(img))
            raw_images.append(img)

            if self.xyz_base is not None:
                img_xyz = Image.open(
                    os.path.join(target_xyz_dir, f"xyz_new_{image_index:03d}.png")
                ).convert("RGBA")
                img_xyz = trans_xyz(img_xyz) if trans_xyz is not None else img_xyz
                img_xyz = trans(img_xyz) if trans is not None else img_xyz
                img_xyz = do_resize_content(img_xyz, cur_resize_rate)
                img_xyz.putalpha(alpha_mask)
                if self.random_background:
                    img_xyz = add_random_background(img_xyz, bg_color)
                img_xyz = img_xyz.convert("RGB")
                target_xyz_images.append(self.transfrom_vae(img_xyz))
                if self.debug:
                    raw_xyz_images.append(img_xyz)

        cameras = [get_camera_for_index(i).squeeze() for i in self.camera_views]
        if self.ref_position is not None:
            cameras[self.ref_position] = torch.zeros_like(
                cameras[self.ref_position]
            )  # set ref camera to zero

        cameras = torch.stack(cameras)

        input_img = Image.open(
            os.path.join(target_dir, f"{ref_index:03d}.png")
        ).convert("RGBA")
        input_img = do_resize_content(input_img, cur_resize_rate)
        if random.random() < self.stroke_p:
            ## random rgb color
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            radius = random.randint(1, 3)
            input_img = add_stroke(input_img, color=color, stroke_radius=radius)
        if self.random_background:
            input_img = add_random_background(input_img, bg_color)
        input_img = input_img.convert("RGB")

        clip_cond = self.transfrom_clip(input_img)
        vae_cond = self.transfrom_vae(input_img)

        vae_target = torch.stack(target_images, dim=0)
        if self.xyz_base is not None:
            xyz_vae_target = torch.stack(target_xyz_images, dim=0)
        else:
            xyz_vae_target = []

        if self.debug:
            print(f"debug!!,{bg_color}")
            return {
                "target_images": raw_images,
                "target_images_xyz": raw_xyz_images,
                "input_img": input_img,
                "cameras": cameras,
                "caption": caption,
                "item": self.items[index],
                "alpha_masks": alpha_masks,
                "cur_resize_rate": cur_resize_rate,
            }

        if self.split == "train":
            return {
                "target_images_vae": vae_target,
                "target_images_xyz_vae": xyz_vae_target,
                "clip_cond": clip_cond,
                "vae_cond": vae_cond,
                "cameras": cameras,
                "caption": caption,
            }
        else:  # eval
            path = os.path.join(target_dir, f"{ref_index:03d}.png")
            return dict(
                path=path,
                target_dir=target_dir,
                cond_raw_images=raw_images,
                cond=input_img,
                ref_index=ref_index,
                ident=f"{index}-{Path(target_dir).stem}",
            )


class InTheWildImages(Dataset):
    """
    a data set for in the wild images,
    receive base floders, image path ls, path files as input
    """

    def __init__(self, base_dirs=[], image_paths=[], path_files=[]):
        print(__class__)
        self.base_dirs = base_dirs
        self.image_paths = image_paths
        self.path_files = path_files
        self.init_item()

    def init_item(self):
        items = []
        for d in self.base_dirs:
            items += [osp.join(d, f) for f in os.listdir(d)]
        items = items + self.image_paths

        for file in self.path_files:
            with open(file, "r") as f:
                items += [line.strip() for line in f.readlines()]
        items.sort()
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img = Image.open(item)
        background = Image.new("RGBA", img.size, (0, 0, 0, 0))
        cond = Image.alpha_composite(background, img)
        return dict(
            path=item, ident=f"{index}-{Path(item).stem}", cond=cond.convert("RGB")
        )
