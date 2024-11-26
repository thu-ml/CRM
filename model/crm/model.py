import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


from pathlib import Path
import cv2
import trimesh
import nvdiffrast.torch as dr

from model.archs.decoders.shape_texture_net import TetTexNet
from model.archs.unet import UNetPP
from util.renderer import Renderer
from util.utils import get_tri, trans_depth
from model.archs.mlp_head import SdfMlp, RgbMlp
import xatlas
import random


class Dummy:
    pass

class CRM(nn.Module):
    def __init__(self, specs):
        super(CRM, self).__init__()

        self.specs = specs
        # configs
        input_specs = specs["Input"]
        self.input = Dummy()
        self.input.scale = input_specs['scale']
        self.input.resolution = input_specs['resolution']
        self.tet_grid_size = input_specs['tet_grid_size']
        self.camera_angle_num = input_specs['camera_angle_num']

        arch_specs = specs["ArchSpecs"]
        self.arch = Dummy()
        self.arch.fea_concat = arch_specs["fea_concat"]
        self.arch.mlp_bias = arch_specs["mlp_bias"]

        encoder_specs = specs["EncoderSpecs"]
        self.enc = Dummy()
        self.enc.plane_resolution = encoder_specs["plane_resolution"]
        decoder_specs = specs["DecoderSpecs"]
        self.dec = Dummy()
        self.dec.c_dim = decoder_specs["c_dim"]
        self.dec.plane_resolution = decoder_specs["plane_resolution"]

        self.geo_type = specs["Train"].get("geo_type", "dmtet") # "dmtet" or "flex"

        self.unet2 = UNetPP(in_channels=self.dec.c_dim)

        mlp_chnl_s = 3 if self.arch.fea_concat else 1  # 3 for queried triplane feature concatenation
        self.decoder = TetTexNet(plane_reso=self.dec.plane_resolution, fea_concat=self.arch.fea_concat)

        if self.geo_type == "flex":
            self.weightMlp = nn.Sequential(
                            nn.Linear(mlp_chnl_s * 32 * 8, 512),
                            nn.SiLU(),
                            nn.Linear(512, 21))         
        self.sdfMlp = SdfMlp(mlp_chnl_s * 32, 512, bias=self.arch.mlp_bias) 
        self.rgbMlp = RgbMlp(mlp_chnl_s * 32, 512, bias=self.arch.mlp_bias)
        self.renderer = Renderer(tet_grid_size=self.tet_grid_size, camera_angle_num=self.camera_angle_num,
                                 scale=self.input.scale, geo_type = self.geo_type)

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.ticker = 0

        self.spob = True if specs['Pretrain']['mode'] is None else False  # whether to add sphere
        self.radius = specs['Pretrain']['radius']  # used when spob

        self.denoising = False
        from diffusers import DDIMScheduler
        self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
    def small_fwd(self, data):
        xyzs_list = []
        color_list = []
        # color0_list = []
        # print(data['triview_color'].shape)
        # print(data['triview_xyz'].shape)
        for i in range(len(data['gt_verts'])):
            color = data['triview_color'][i].permute(0,3,1,2).clone()
            xyzs = data['triview_xyz'][i].permute(0,3,1,2).clone()

            if xyzs.max() > 1.1:
                xyzs = xyzs / 255
                # print("fix 255 bug")
            if self.specs["Train"].get("random_bg", False):
                crgb = 0.5#random.random()
                cxyz = 0.5#random.random()
                # for j in range(6):
                #     mask = torch.mean(xyzs[j], dim = 0, keepdim=False)
                #     mask = (mask == 0)
                #     grey_image = torch.zeros_like(cxyz)
                #     for k in range(3):
                        # grey_image[j][k][mask] = cxyz
                for j in range(6):
                    mask = torch.mean(xyzs[j], dim = 0, keepdim=False)
                    mask = (mask == 0)
                    for k in range(3):               
                        color[j][k][mask] = crgb
                        xyzs[j][k][mask] = cxyz
            else:
                crgb = 0
                cxyz = 0
                
            shift = 0
            if random.random() < 0.5:
                shift = int(self.specs["Train"].get("shift", 0))
            # print(color.shape)
            color_tensor1 = get_tri(color, dim=0, blender=True, c=crgb, shift=shift, rgb=True)
            xyzs_tensor1 = get_tri(xyzs, dim=0, blender=True, c=cxyz, shift=shift, fix=True) # , fix = True
            # print(color_tensor1.min(), color_tensor1.max(), xyzs_tensor1.min(),xyzs_tensor1.max())

            
            # print(color_tensor1.shape)
            color_list.append(color_tensor1)
            xyzs_list.append(xyzs_tensor1)
        color_tensor = torch.stack(color_list, dim=0)
        # print(color_tensor.shape)
        color_tensor = torch.nn.functional.interpolate(color_tensor, [self.enc.plane_resolution, self.enc.plane_resolution*3], mode = "bilinear")

        xyzs_tensor = torch.stack(xyzs_list, dim=0)
        xyzs_tensor = torch.nn.functional.interpolate(xyzs_tensor, [self.enc.plane_resolution, self.enc.plane_resolution*3], mode = "bilinear")

        # if random.random()<0.5:
        #     color_tensor = torch.nn.functional.interpolate(color_tensor, [256, 256*3], mode = "bilinear")
            # xyzs_tensor = torch.nn.functional.interpolate(xyzs_tensor, [256, 256*3], mode = "bilinear")
        #     color_tensor = torch.nn.functional.interpolate(color_tensor, [self.enc.plane_resolution, self.enc.plane_resolution*3], mode = "bilinear")
            # xyzs_tensor = torch.nn.functional.interpolate(xyzs_tensor, [self.enc.plane_resolution, self.enc.plane_resolution*3], mode = "bilinear")
        
        # reso_tmp = random.randint(32, 256)
        # xyzs_tensor = torch.nn.functional.interpolate(xyzs_tensor, [reso_tmp, reso_tmp*3], mode = "bilinear")
        # xyzs_tensor = torch.nn.functional.interpolate(xyzs_tensor, [self.enc.plane_resolution, self.enc.plane_resolution*3], mode = "bilinear")
        
        # !Notice
        all_tensor = torch.cat([color_tensor, xyzs_tensor], dim = 1)# *2 - 1
        # print(all_tensor.shape)
        triplane_feature0 = all_tensor
        triplane_feature1 = all_tensor
        return triplane_feature0, triplane_feature1
                 
    def forward(self, data, ctx, tnew=None):
        # pcd_xyz = data['gt_pcd']
        # pcd_rgb = data['gt_pcd_colors']
        # dense_pcd_xyz = data['gt_pcd_query']
        # dense_pcd_rgb = data['gt_pcd_colors_query']

        # print(data['triview_color'].shape)
        # print(data['triview_xyz'].shape)
        # CYF Note: Adding noise to xyz. In real applications,
        # point cloud is generated by sensors and thus noisy
        # pcd_xyz += xyz_noise_scale * torch.randn(pcd_xyz.shape).to(pcd_xyz.get_device())
        # dense_pcd_xyz += xyz_noise_scale * torch.randn(dense_pcd_xyz.shape).to(dense_pcd_xyz.get_device())

        # print(verts.shape, tets.shape)
        # print(self.renderer.flexicubes.verts.shape, self.renderer.flexicubes.indices.shape)
        if self.geo_type == "flex":
            # use the number of data not batch size
            verts = self.renderer.flexicubes.verts.unsqueeze(0).repeat(len(data['gt_verts']), 1, 1)
            tets = self.renderer.flexicubes.indices

        # if self.arch.tri_img:
        with torch.no_grad():
            tri_fea_0, tri_fea_1 = self.small_fwd(data)

        # if self.denoising == True:
        #     # tnew = 250
        #     if tnew == None:
        #         tnew = torch.randint(0, 1000, [tri_fea_1.shape[0]], dtype=torch.long, device=tri_fea_1.device)
        #     else:
        #         tnew = torch.randint(tnew, tnew+1, [tri_fea_1.shape[0]], dtype=torch.long, device=tri_fea_1.device)
        #     noise_new = torch.randn_like(tri_fea_1) *0.5+0.5
        #     reso = tri_fea_1.shape[-2]
        #     cache1 = tri_fea_1[:,:3,:,reso*2:]
        #     tri_fea_1 =self.scheduler.add_noise(tri_fea_1, noise_new, tnew)
        #     tri_fea_1[:,:3,:,reso*2:] = cache1

        # if self.arch.tri_img or self.arch.tri_mix:
        # if self.denoising == True:
        #     tri_fea_2 = self.unet2(tri_fea_1, tnew)
        # else:
        tri_fea_2 = self.unet2(tri_fea_1)
        # else:
        #     reso = tri_fea_1.shape[-2:]
        #     reso_tmp = random.randint(64, 256)#int(torch.randint(low=64, high=256, size=(1,), dtype=torch.int))    
        #     tri_fea_1 = torch.nn.functional.interpolate(tri_fea_1, size=(reso_tmp, reso_tmp * 3), mode='bilinear')
        #     # tri_fea_1 = tri_fea_1.clamp(-1,1)
        #     # KL here
        #     tri_fea_1 = torch.nn.functional.interpolate(tri_fea_1, size=reso, mode='bilinear')
            
        #     tri_fea_2 = self.unet2(tri_fea_1)

        assert torch.isnan(tri_fea_2).sum() == 0, print("nan in triplane feature 2")
        dec_verts = self.decoder(tri_fea_2, verts)

        pred_shape = self.sdfMlp(dec_verts)
        assert torch.isnan(pred_shape).sum() == 0, print("nan in predicted shape primitives")

        pred_sdf, deformation = pred_shape[..., 0], pred_shape[..., 1:4]
        if self.spob:
            pred_sdf = pred_sdf + self.radius - torch.sqrt((verts ** 2).sum(-1))

        for i in range(pred_sdf.shape[0]):
            if pred_sdf[i].min() >= 0 or pred_sdf[i].max() <= 0:
                print('004occur!', pred_sdf[i].max(), pred_sdf[i].min())
                pred_sdf[i] -= (pred_sdf[i].min().detach() / 8 + pred_sdf[i].max().detach() * 7 / 8)

        weight = None
        if self.geo_type == "flex":
            grid_feat = torch.index_select(input=dec_verts, index=self.renderer.flexicubes.indices.reshape(-1),dim=1)
            grid_feat = grid_feat.reshape(dec_verts.shape[0], self.renderer.flexicubes.indices.shape[0], self.renderer.flexicubes.indices.shape[1] * dec_verts.shape[-1])
            weight = self.weightMlp(grid_feat)
            weight = weight * 0.1
            # print("weight shape", weight.shape)
        # Debug HOLD: SDF output
        # vis = out.detach().cpu().numpy()
        # print(pred_sdf.min(), pred_sdf.max(), (pred_sdf <= 0).sum(), (pred_sdf > 0).sum())

        # valid = True
        # for i in range(len(data['gt_verts'])):
        #     xyzs = data['triview_xyz'][i].permute(0,3,1,2)
        #     ma1 = 255
        #     ma2 = 255
        #     mb1 = 0
        #     mb2 = 0
        #     for j in range(6):
        #         a1,a2,b1,b2 = find_bounding_box(torch.mean(xyzs[j], dim = 0, keepdim=False))
        #         ma1 = min(a1, ma1)
        #         ma2 = min(a2, ma2)
        #         mb1 = max(b1, mb1)
        #         mb2 = max(b2, mb2)
        #     # print(min(ma1, ma2), max(mb1, mb2))
        #     if min(ma1, ma2) != 0 or max(mb1, mb2) != 255:
        #         valid = False
        
        # valid = True
        
        result, verts, faces = self.renderer(data, pred_sdf, deformation, verts, tets, training = self.training, weight = weight)
        # if not valid:
        #     print("valid is false")
        #     result['silhouette_loss'] *= 0
        #     result['depth_loss'] *= 0

        sdf_pos = pred_sdf[pred_sdf>=0]
        sdf_neg = pred_sdf[pred_sdf<0]
        result["wzy_loss"] = (torch.abs(sdf_pos-1).sum() + torch.abs(sdf_neg+1).sum()) / (pred_sdf.shape[0]*pred_sdf.shape[1])

        # if self.tex_sup_mode == "pcd-raw":
        #     dense_pcd_xyz = data['gt_pcd_query']
        #     dense_pcd_rgb = data['gt_pcd_colors_query']
        #     # print("raw texture supervision")
        #     dec_d_pcd = self.decoder(tri_fea_2, dense_pcd_xyz)
        #     pred_rgb = self.rgbMlp(dec_d_pcd)  # torch.cat((dec_d_pcd, dense_pcd_xyz), dim=2)
        #     color_loss = self.l1_loss(pred_rgb, dense_pcd_rgb) + self.mse_loss(pred_rgb, dense_pcd_rgb)

        # elif self.tex_sup_mode == "pcd-nnb":
        #     dense_pcd_xyz = data['gt_pcd_query']
        #     dense_pcd_rgb = data['gt_pcd_colors_query']
        #     # print("nearest neighbor supervision")
        #     with torch.no_grad():
        #         pc_list = []
        #         for i in range(len(verts)):
        #             pred_points = kaolin.ops.mesh.sample_points(verts[i].unsqueeze(0), faces[i], dense_pcd_xyz.shape[1])[0][0]
        #             # print(pred_points.shape)
        #             pc_list.append(pred_points.unsqueeze(0))
        #         pcs = torch.cat(pc_list, dim=0)
        #         # pcs += noise_scale * torch.randn(pcs.shape).to(pcs.get_device())  # pcs collected from model, no need to add noise
        #         pcs_color = find_nearest_points(pcs, dense_pcd_xyz, dense_pcd_rgb)

        #     dec_d_pcd = self.decoder(tri_fea_2, pcs)
        #     pred_rgb = self.rgbMlp(dec_d_pcd)
        #     color_loss = self.l1_loss(pred_rgb, pcs_color) + self.mse_loss(pred_rgb, pcs_color)

        # if self.tex_sup_mode == 'blender':
        # if self.mvdream==False and (self.sd_mode == "sds" or self.sd_mode == "vsd"):
        #     resolution = [512,512]
        # else:
        resolution = data['resolution_img']
        gt_color = data['gt_render_imgs']
        result['gt_render_imgs'] = data['gt_render_imgs']
        # print(data["prompt"])
        img_list = []
        # if self.sd_mode == "sds" or self.sd_mode == "vsd":
        #     latents_list = []
        #     latents_gt_list = []
        #     img_list2 = []
        #     if self.sd_mode == "vsd":
        #         latents_train_list = []
        #         latents_train_gt_list = []
        # sds_bs = 4
        mvp_size = data['mvp'][0].shape[0]
        for i in range(len(verts)):
            mvp = data['mvp'][i]
            # print(mvp.shape, data['mvp'].shape)
            # if self.sd_mode == "sds" or self.sd_mode == "vsd":
            #     azimuth = 90*random.random()
            #     elevation = 30*random.random()
            #     fov = 15+(60-15)*random.random()
            #     mvp2 = sds_scene_mv(sds_bs, azimuth, elevation, fov).to(data['mvp'][i].device)
            #     mvp = torch.cat([mvp, mvp2], dim = 0)
            #     # print(mvp.shape)

            face = faces[i].int()

            verts_clip = torch.bmm(
                F.pad(verts[i], pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1),
                mvp.permute(0, 2, 1)).float()  # [B, N, 4]
            rast, rast_db = dr.rasterize(ctx, verts_clip, face, resolution)
            # print(verts.shape, faces.shape)
            alpha, _ = dr.interpolate(torch.ones_like(verts[i][:, :1]).unsqueeze(0), rast, face)  # [1, H, W, 1]
            xyzs, _ = dr.interpolate(verts[i], rast, face)  # [1, H, W, 3]
            # print(alpha.shape, xyzs.shape)
            xyzs = xyzs.view(-1, 3)
            mask = (alpha > 0).view(-1).detach()
            # do the lighting here since we have normal from mesh now.
            albedo = torch.zeros_like(xyzs, dtype=torch.float32)

            albedo = albedo.view(-1, resolution[0], resolution[1], 3)
            albedo[:mvp_size] = albedo[:mvp_size] - 1
            albedo = albedo.view(-1, 3)
            
            # print("4.0:{}".format(torch.cuda.memory_allocated(0)))
            if mask.any():
                # print(mask.sum())
                dec_d_pcd = self.decoder(tri_fea_2[i].unsqueeze(0), xyzs[mask].unsqueeze(0))
                # print("4.1:{}".format(torch.cuda.memory_allocated(0)))
                masked_albedo = self.rgbMlp(dec_d_pcd)
                # print("4.2:{}".format(torch.cuda.memory_allocated(0)))
                # print(albedo.shape, masked_albedo.shape, mask.shape, albedo.shape)
                albedo[mask] = masked_albedo.squeeze(0).float()
                # print("4.3:{}".format(torch.cuda.memory_allocated(0)))
                del masked_albedo
            albedo = albedo.view(-1, resolution[0], resolution[1], 3)
            albedo = (albedo * 0.5 + 0.5).clip(0, 1)
            # print("4.4:{}".format(torch.cuda.memory_allocated(0)))
            # if self.mvdream==False and (self.sd_mode == "sds" or self.sd_mode == "vsd"):
            #     img_tmp = torch.nn.functional.interpolate(albedo.permute(0,3,1,2)[:mvp_size], [256,256], mode="bilinear").permute(0,2,3,1)
            #     img_list.append(img_tmp)
            # else:
            img_list.append(albedo[:mvp_size])
            # if self.sd_mode == "sds" or self.sd_mode == "vsd":
            #     img_list2.append(albedo[mvp_size:])
            
            # if self.sd_mode == "sds" or self.sd_mode == "vsd":
            #     # print(data["prompt"][i])
            #     cur_prompt = data["prompt"][i]
                
            #     latents = torch.nn.functional.interpolate(albedo.permute(0,3,1,2)[mvp_size:], [512,512] if self.mvdream == False else [256,256], mode="bilinear")
            #     latents = self.unet_sd.encode_imgs(latents)
            #     if self.mvdream == True:
            #         t = torch.randint(20, 980 + 1, [1], dtype=torch.long, device=latents.device)
            #         t = t.repeat(latents.shape[0])
            #     else:
            #         t = torch.randint(20, 980 + 1, [latents.shape[0]], dtype=torch.long, device=latents.device)
            #     noise = torch.randn_like(latents)
            #     latents_noisy = self.unet_sd.scheduler.add_noise(latents, noise, t)
            #     with torch.no_grad() if self.jacobi == False else torch.enable_grad():
            #         if self.mvdream == True:
            #             camera = get_camera(4, elevation, azimuth).to(latents.device)
            #             uncond = {
            #                 "context": self.mv_model.get_learned_conditioning([""]*4).to(latents.device),
            #                 "camera": camera,
            #                 "num_frames": 4,
            #             }
            #             cond = {
            #                 "context": self.mv_model.get_learned_conditioning([cur_prompt[0]]*4).to(latents.device),
            #                 "camera": camera,
            #                 "num_frames": 4,
            #             }
            #             noise_pred = self.mv_model.apply_model(latents_noisy, t, cond=uncond)
            #             noise_pred_cond = self.mv_model.apply_model(latents_noisy, t, cond=cond)
            #         else:
            #             noise_pred = self.unet_sd(latents_noisy, t, "") 
            #             noise_pred_cond = self.unet_sd(latents_noisy, t, cur_prompt)# cur_prompt                                      
            #         if self.sd_mode == "sds":
            #             if self.mvdream == True:
            #                 noise_cfg = noise_pred + self.specs["Train"].get("sd_cfg", 50) * (noise_pred_cond - noise_pred)
            #             else:
            #                 noise_cfg = noise_pred + self.specs["Train"].get("sd_cfg", 100) * (noise_pred_cond - noise_pred)
            #         elif self.sd_mode == "vsd":
            #             noise_pred_lora = self.unet_lora(latents_noisy, t, self.unet_sd.encode_prompt(latents_noisy, cur_prompt)).sample
            #             noise_cfg = noise_pred + self.specs["Train"].get("sd_cfg", 7.5) * (noise_pred_cond - noise_pred) + noise - noise_pred_lora
            #         alpha_prod_t = self.unet_sd.scheduler.alphas_cumprod.to(latents.device)[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            #         beta_prod_t = (1 - alpha_prod_t).to(latents.device)
            #         pred_original_sample = (latents_noisy - beta_prod_t ** (0.5) * noise_cfg) / alpha_prod_t ** (0.5)
            #         # cfg_rescale = True
            #         # if cfg_rescale == False:
            #         #     pred_original_sample = (latents_noisy - beta_prod_t ** (0.5) * noise_cfg) / alpha_prod_t ** (0.5)
            #         # else:
            #         #     pass
            #     latents_list.append(latents if self.jacobi == False else latents.detach())#*alpha_prod_t ** (0.25)
            #     latents_gt_list.append(pred_original_sample.detach() if self.jacobi == False else pred_original_sample)#*alpha_prod_t ** (0.25)

            #     if self.sd_mode == "vsd": #Train LoRA
            #         t_lora = torch.randint(20, 980 + 1, [latents.shape[0]], dtype=torch.long, device=latents.device)
            #         noise_lora = torch.randn_like(latents)
            #         latents_noisy_lora = self.unet_sd.scheduler.add_noise(latents.detach(), noise_lora, t_lora)      
            #         noise_pred_lora_train = self.unet_lora(latents_noisy_lora, t_lora, self.unet_sd.encode_prompt(latents_noisy, cur_prompt)).sample                  
            #         latents_train_list.append(noise_pred_lora_train)
            #         latents_train_gt_list.append(noise_lora)
            del albedo

        # print("5:{}".format(torch.cuda.memory_allocated(0)))
        imgs = torch.stack(img_list, dim=0)
        result['albedo_list'] = imgs
        # if self.sd_mode == "sds" or self.sd_mode == "vsd":
        #     imgs2 = torch.stack(img_list2, dim=0)
        #     result['albedo_list2'] = imgs2

        #     latents_imgs = torch.stack(latents_list, dim=0)
        #     latents_gt_imgs = torch.stack(latents_gt_list, dim=0)
        #     color_loss = 0.5 * self.mse_loss(latents_imgs, latents_gt_imgs) * self.specs["Train"]["sd_lambda"] #+ self.mse_loss(imgs, gt_color)
        #     if self.sd_mode == "vsd":
        #         latents_train_imgs = torch.stack(latents_train_list, dim=0)
        #         latents_train_gt_imgs = torch.stack(latents_train_gt_list, dim=0)
        #         color_loss = color_loss + self.mse_loss(latents_train_imgs, latents_train_gt_imgs) # Added but optimizing different params
        #     # color_loss = self.l1_loss(imgs, gt_color) + self.mse_loss(imgs, gt_color) + color_loss
        # else:
            # print(data['alpha_ctn'].min(),data['alpha_ctn'].max())
        color_loss = self.mse_loss(imgs, gt_color) # self.l1_loss(imgs, gt_color) + 
            # if not valid:
            #     color_loss *= 0
        # print("6:{}".format(torch.cuda.memory_allocated(0)))
        result.update({
            'color_gt': gt_color
        })

        result['color_loss'] = color_loss
        result["reg_loss"] = torch.norm(tri_fea_1)

        result["tri0"], result["tri1"], result["tri2"] = tri_fea_0, tri_fea_1, tri_fea_2

        return result, verts, faces

    def decode(self, data, triplane_feature2):
        if self.geo_type == "flex":
            tet_verts = self.renderer.flexicubes.verts.unsqueeze(0)
            tet_indices = self.renderer.flexicubes.indices

        dec_verts = self.decoder(triplane_feature2, tet_verts)
        out = self.sdfMlp(dec_verts)

        weight = None
        if self.geo_type == "flex":
            grid_feat = torch.index_select(input=dec_verts, index=self.renderer.flexicubes.indices.reshape(-1),dim=1)
            grid_feat = grid_feat.reshape(dec_verts.shape[0], self.renderer.flexicubes.indices.shape[0], self.renderer.flexicubes.indices.shape[1] * dec_verts.shape[-1])
            weight = self.weightMlp(grid_feat)
            weight = weight * 0.1

        pred_sdf, deformation = out[..., 0], out[..., 1:]
        if self.spob:
            pred_sdf = pred_sdf + self.radius - torch.sqrt((tet_verts**2).sum(-1))

        _, verts, faces = self.renderer(data, pred_sdf, deformation, tet_verts, tet_indices, weight= weight)
        return verts[0].unsqueeze(0), faces[0].int()
    @torch.no_grad()
    def render_color_img(self, verts, faces, data, ctx, gt=False, mvp=None, idx = 0):
        mvp = data['mvp'][idx] if mvp is None else mvp[0]
        res = data['resolution_img']
        # print(mvp.shape, verts.shape, faces.shape, res)
        faces = faces.int()
        faces = faces[..., [2, 1, 0]]
        # rasterization
        verts_clip = torch.bmm(
            F.pad(verts, pad=(0, 1), mode='constant', value=1.0).repeat(mvp.shape[0], 1, 1),
            mvp.permute(0, 2, 1)).float()  # [B, N, 4]
        # assert verts_clip.shape[0] > 0 and verts_clip.shape[1] > 0, print(verts_clip.shape, verts.shape)
        rast, _ = dr.rasterize(ctx, verts_clip, faces, res)
        
        # print(verts.shape, faces.shape)
        alpha, _ = dr.interpolate(torch.ones_like(verts[0][:, :1]).unsqueeze(0), rast, faces) # [1, H, W, 1]
        xyzs, _ = dr.interpolate(verts[0], rast, faces) # [1, H, W, 3]
        # print(alpha.shape, xyzs.shape)
        xyzs = xyzs.view(-1, 3)
        mask = (alpha > 0).view(-1).detach()
        # do the lighting here since we have normal from mesh now.
        albedo = torch.zeros_like(xyzs, dtype=torch.float32) - 1
        if mask.any():
            # if gt==False:
            # if self.arch.tri_img:
            _, triplane_feature1 = self.small_fwd(data)
            #     elif self.arch.tri_mix:
            #         triplane_feature0_a, triplane_feature1_a = self.small_fwd(data)
            #         pc = data['gt_pcd'][idx].unsqueeze(0)
            #         pc_colors = data['gt_pcd_colors'][idx].unsqueeze(0)
            #         triplane_feature0_b = self.encoder(pc, pc_colors)
            #         triplane_feature1_b = self.unet1(triplane_feature0_b)  # unet1 moved to trdgen main class
            #         reso_ = triplane_feature0_a.shape[-2]
            #         triplane_feature0_b = torch.nn.functional.interpolate(triplane_feature0_b, size=(reso_, reso_ * 3), mode='bilinear')
            #         triplane_feature1_b = torch.nn.functional.interpolate(triplane_feature1_b, size=(reso_, reso_ * 3), mode='bilinear')
            #         triplane_feature0 = torch.cat([triplane_feature0_a,triplane_feature0_b],dim=1)
            #         triplane_feature1 = torch.cat([triplane_feature1_a,triplane_feature1_b],dim=1)
            #     else:
            #         pc = data['gt_pcd'][idx].unsqueeze(0)
            #         pc_colors = data['gt_pcd_colors'][idx].unsqueeze(0)
            #         triplane_feature0 = self.encoder(pc, pc_colors)
            #         triplane_feature1 = self.unet1(triplane_feature0)  # unet1 moved to trdgen main class
            #         assert torch.isnan(triplane_feature1).sum() == 0, print("nan in tri")
            triplane_feature2 = self.unet2(triplane_feature1)  # unet2 moved to trdgen main class
            masked_albedo = self.rgbMlp(self.decoder(triplane_feature2, xyzs[mask].unsqueeze(0))).squeeze(0)
            # else:
            #     masked_albedo = find_nearest_points(xyzs[mask].unsqueeze(0), data['gt_pcd_query'][idx].unsqueeze(0), data['gt_pcd_colors_query'][idx].unsqueeze(0)).squeeze(0)
            albedo[mask] = masked_albedo.float()
        albedo = albedo.view(-1, alpha.shape[1], alpha.shape[2], 3)
        albedo = (albedo * 0.5 + 0.5).clip(0, 1)

        mask2 = (alpha <= 0).view(-1).detach()
        xyzs[mask2] -= 1
        xyzs = xyzs.view(alpha.shape[0], alpha.shape[1], alpha.shape[2], 3)
        xyzs = (xyzs * 0.5 + 0.5).clip(0, 1)
        # print(albedo.shape)
        return albedo, xyzs

    def visualize(self, result, epc, chunk, color_fine, rdr_img_dir, iter_all):
        # After each epoch, visualize and evaluate
        # epc: current index of epochs
        normal = result['normal_image'][0].detach().cpu().numpy()
        normal_gt = result['normal_gt'][0].detach().cpu().numpy()
        alpha = (result['alpha'][0].detach().cpu().numpy() * 255).astype('uint8')
        alpha_gt = (result['alpha_gt'][0].detach().cpu().numpy() * 255).astype('uint8')

        succ = True
        try:
            depth = trans_depth(result['depth'])
            depth_gt = trans_depth(result['depth_gt'])
        except:
            succ = False
        color_fine = color_fine[..., [2, 1, 0]].detach().cpu().numpy()
        color_fine = cv2.resize(color_fine[0], alpha.shape[1:3])[None, ...]
        # if result.get('color_gt') is None:
        #     # color_gt = color_gt[..., [2, 1, 0]].detach().cpu().numpy()
        #     color_fine_gt = color_fine_gt[..., [2, 1, 0]].detach().cpu().numpy()
        # else:
        color_fine_gt = result.get('color_gt')[0, ..., [2, 1, 0]].cpu().numpy()
        color_fine_gt = cv2.resize(color_fine_gt[0], alpha.shape[1:3])[None, ...]
        if succ == True:
            img = np.concatenate((alpha, alpha_gt, depth, depth_gt), axis=2)
        else:
            img = np.concatenate((alpha, alpha_gt, alpha, alpha_gt), axis=2)
        if succ == True:
            cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}.png", np.concatenate((normal[0], color_fine[0], color_fine_gt[0], cv2.applyColorMap(np.repeat(depth[0],3,axis=2).astype(np.uint8), cv2.COLORMAP_HOT)/255, cv2.applyColorMap(np.repeat(depth_gt[0],3,axis=2).astype(np.uint8), cv2.COLORMAP_HOT)/255), axis=1)*255)
        else:
            cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}.png", np.concatenate((normal[0], color_fine[0], color_fine_gt[0]), axis=1)*255)
        
        
        tri0 = result['tri0'][0].mean(0).unsqueeze(-1)
        tri0 = (tri0 - tri0.min()) / (tri0.max() - tri0.min())
        tri0 = tri0.detach().cpu().numpy()
        tri0 = np.repeat(tri0, 3, axis=2) * 255

        tri1 = result['tri1'][0].mean(0).unsqueeze(-1)
        tri1 = (tri1 - tri1.min()) / (tri1.max() - tri1.min())
        tri1 = tri1.detach().cpu().numpy()
        tri1 = np.repeat(tri1, 3, axis=2) * 255

        tri2 = result['tri2'][0].mean(0).unsqueeze(-1)
        tri2 = (tri2 - tri2.min()) / (tri2.max() - tri2.min())
        tri2 = tri2.detach().cpu().numpy()
        tri2 = np.repeat(tri2, 3, axis=2) * 255

        cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}-tri-mean.jpg", cv2.applyColorMap(np.concatenate((tri0, tri1, tri2), axis=0).astype(np.uint8), cv2.COLORMAP_JET))

        self.ticker += 1

        i0 = self.ticker % result['tri0'][0].shape[0]
        i1 = self.ticker % result['tri1'][0].shape[0]
        i2 = self.ticker % result['tri2'][0].shape[0]

        tri0 = result['tri0'][0][i0].unsqueeze(-1)
        tri0 = (tri0-tri0.min())/(tri0.max()-tri0.min())
        tri0 = tri0.detach().cpu().numpy()
        tri0 = np.repeat(tri0, 3, axis=2) * 255

        tri1 = result['tri1'][0][i1].unsqueeze(-1)
        tri1 = (tri1-tri1.min())/(tri1.max()-tri1.min())
        tri1 = tri1.detach().cpu().numpy()
        tri1 = np.repeat(tri1, 3, axis=2) * 255

        tri2 = result['tri2'][0][i2].unsqueeze(-1)
        tri2 = (tri2-tri2.min())/(tri2.max()-tri2.min())
        tri2 = tri2.detach().cpu().numpy()
        tri2 = np.repeat(tri2, 3, axis=2) * 255

        cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}-tri{i0}-{i1}-{i2}.jpg", cv2.applyColorMap(np.concatenate((tri0, tri1, tri2), axis=0).astype(np.uint8), cv2.COLORMAP_JET))

        return img

    def export_mesh(self, data, out_dir, ind, device=None, tri_fea_2 = None):
        verts = data['verts']
        faces = data['faces']

        dec_verts = self.decoder(tri_fea_2, verts.unsqueeze(0))
        colors = self.rgbMlp(dec_verts).squeeze().detach().cpu().numpy()
        # Expect predicted colors value range from [-1, 1]
        colors = (colors * 0.5 + 0.5).clip(0, 1)

        verts = verts.squeeze().cpu().numpy()
        faces = faces[..., [2, 1, 0]].squeeze().cpu().numpy()

        # export the final mesh
        with torch.no_grad():
            mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors, process=False) # important, process=True leads to seg fault...
            mesh.export(out_dir / f'{ind}.obj')

    def export_mesh_wt_uv(self, ctx, data, out_dir, ind, device, res, tri_fea_2=None):
        mesh_v = data['verts'].squeeze().cpu().numpy()
        mesh_pos_idx = data['faces'].squeeze().cpu().numpy()

        def interpolate(attr, rast, attr_idx, rast_db=None):
            return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db,
                                  diff_attrs=None if rast_db is None else 'all')

        vmapping, indices, uvs = xatlas.parametrize(mesh_v, mesh_pos_idx)

        mesh_v = torch.tensor(mesh_v, dtype=torch.float32, device=device)
        mesh_pos_idx = torch.tensor(mesh_pos_idx, dtype=torch.int64, device=device)

        # Convert to tensors
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

        uvs = torch.tensor(uvs, dtype=torch.float32, device=mesh_v.device)
        mesh_tex_idx = torch.tensor(indices_int64, dtype=torch.int64, device=mesh_v.device)
        # mesh_v_tex. ture
        uv_clip = uvs[None, ...] * 2.0 - 1.0

        # pad to four component coordinate
        uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., 0:1]), torch.ones_like(uv_clip[..., 0:1])), dim=-1)

        # rasterize
        rast, _ = dr.rasterize(ctx, uv_clip4, mesh_tex_idx.int(), res)

        # Interpolate world space position
        gb_pos, _ = interpolate(mesh_v[None, ...], rast, mesh_pos_idx.int())
        mask = rast[..., 3:4] > 0

        # return uvs, mesh_tex_idx, gb_pos, mask
        gb_pos_unsqz = gb_pos.view(-1, 3)
        mask_unsqz = mask.view(-1)
        tex_unsqz = torch.zeros_like(gb_pos_unsqz) + 1

        gb_mask_pos = gb_pos_unsqz[mask_unsqz]

        gb_mask_pos = gb_mask_pos[None, ]

        with torch.no_grad():
            dec_verts = self.decoder(tri_fea_2, gb_mask_pos)
            colors = self.rgbMlp(dec_verts).squeeze()

        # Expect predicted colors value range from [-1, 1]
        lo, hi = (-1, 1)
        colors = (colors - lo) * (255 / (hi - lo))
        colors = colors.clip(0, 255)

        tex_unsqz[mask_unsqz] = colors

        tex = tex_unsqz.view(res + (3,))

        verts = mesh_v.squeeze().cpu().numpy()
        faces = mesh_pos_idx[..., [2, 1, 0]].squeeze().cpu().numpy()
        # faces = mesh_pos_idx
        # faces = faces.detach().cpu().numpy()
        # faces = faces[..., [2, 1, 0]]
        indices = indices[..., [2, 1, 0]]

        # xatlas.export(f"{out_dir}/{ind}.obj", verts[vmapping], indices, uvs)
        matname = f'{out_dir}.mtl'
        # matname = f'{out_dir}/{ind}.mtl'
        fid = open(matname, 'w')
        fid.write('newmtl material_0\n')
        fid.write('Kd 1 1 1\n')
        fid.write('Ka 1 1 1\n')
        # fid.write('Ks 0 0 0\n')
        fid.write('Ks 0.4 0.4 0.4\n')
        fid.write('Ns 10\n')
        fid.write('illum 2\n')
        fid.write(f'map_Kd {out_dir.split("/")[-1]}.png\n')
        fid.close()

        fid = open(f'{out_dir}.obj', 'w')
        # fid = open(f'{out_dir}/{ind}.obj', 'w')
        fid.write('mtllib %s.mtl\n' % out_dir.split("/")[-1])

        for pidx, p in enumerate(verts):
            pp = p
            fid.write('v %f %f %f\n' % (pp[0], pp[2], - pp[1]))

        for pidx, p in enumerate(uvs):
            pp = p
            fid.write('vt %f %f\n' % (pp[0], 1 - pp[1]))

        fid.write('usemtl material_0\n')
        for i, f in enumerate(faces):
            f1 = f + 1
            f2 = indices[i] + 1
            fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
        fid.close()

        img = np.asarray(tex.data.cpu().numpy(), dtype=np.float32)
        mask = np.sum(img.astype(float), axis=-1, keepdims=True)
        mask = (mask <= 3.0).astype(float)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        img = img * (1 - mask) + dilate_img * mask
        img = img.clip(0, 255).astype(np.uint8)

        cv2.imwrite(f'{out_dir}.png', img[..., [2, 1, 0]])
