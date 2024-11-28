import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import cv2
import trimesh
import nvdiffrast.torch as dr
import xatlas
# from diffusers import DDIMScheduler

from model.archs.decoders.shape_texture_net import TetTexNet
from model.archs.unet import UNetPP
from util.renderer import Renderer
from util.utils import get_tri, trans_depth
from model.archs.mlp_head import SdfMlp, RgbMlp, WeightMlp


class Dummy:
    pass

class CRM(nn.Module):
    def __init__(self, specs):
        super().__init__()

        self.specs = specs
        # configs
        input_specs = specs["Input"]
        self.input = Dummy()
        self.input.scale = input_specs['scale']
        self.input.resolution = input_specs['resolution']
        self.tet_grid_size = input_specs['tet_grid_size']
        self.camera_angle_num = input_specs['camera_angle_num']

        # arch_specs = specs["ArchSpecs"]
        # self.arch = Dummy()
        # self.arch.fea_concat = arch_specs["fea_concat"]
        # self.arch.mlp_bias = arch_specs["mlp_bias"]

        encoder_specs = specs["EncoderSpecs"]
        self.enc = Dummy()
        self.enc.plane_resolution = encoder_specs["plane_resolution"]
        decoder_specs = specs["DecoderSpecs"]
        self.dec = Dummy()
        self.dec.c_dim = decoder_specs["c_dim"]
        self.dec.plane_resolution = decoder_specs["plane_resolution"]

        # self.geo_type = specs["Train"].get("geo_type", "dmtet") # "dmtet" or "flex"

        self.unet = UNetPP(in_channels=self.dec.c_dim)

        # mlp_chnl_s = 1  # 3 for queried triplane feature concatenation
        self.decoder = TetTexNet(plane_reso=self.dec.plane_resolution)

        # if self.geo_type == "flex":
        self.weight_mlp = WeightMlp(self.dec.c_dim * 8)
        self.sdf_mlp = SdfMlp(self.dec.c_dim)
        self.rgb_mlp = RgbMlp(self.dec.c_dim)
        self.renderer = Renderer(tet_grid_size=self.tet_grid_size,
                                 camera_angle_num=self.camera_angle_num,
                                 scale=self.input.scale)

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.ticker = 0

        # self.spob = True  # whether to add sphere
        self.radius = specs['Train']['radius']  # used when spob
        # self.denoising = False

        # self.scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")

    def small_fwd(self, data):
        xyzs_list = []
        color_list = []
        for i in range(len(data['verts'])):
            color = data['triview_color'][i].permute(0,3,1,2).clone()
            xyzs = data['triview_xyz'][i].permute(0,3,1,2).clone()

            if xyzs.max() > 1.1:
                xyzs = xyzs / 255
                # print("fix 255 bug")
            # if self.specs["Train"].get("random_bg", False):
            #     crgb = 0.5#random.random()
            #     cxyz = 0.5#random.random()
            #     for j in range(6):
            #         mask = torch.mean(xyzs[j], dim = 0, keepdim=False)
            #         mask = (mask == 0)
            #         for k in range(3):
            #             color[j][k][mask] = crgb
            #             xyzs[j][k][mask] = cxyz
            # else:
            # crgb = 0
            # cxyz = 0

            # shift = 0
            # if random.random() < 0.5:
            #     shift = int(self.specs["Train"].get("shift", 0))

            color_tensor1 = get_tri(color, dim=0, blender=True, rgb=True)
            xyzs_tensor1 = get_tri(xyzs, dim=0, blender=True, fix=True) # , fix = True

            # print(color_tensor1.shape)
            # print(color_tensor1.shape)
            color_list.append(color_tensor1)
            xyzs_list.append(xyzs_tensor1)
        color_tensor = torch.stack(color_list, dim=0)
        color_tensor = torch.nn.functional.interpolate(
            color_tensor, [self.enc.plane_resolution, self.enc.plane_resolution*3], mode = "bilinear")

        xyzs_tensor = torch.stack(xyzs_list, dim=0)
        xyzs_tensor = torch.nn.functional.interpolate(
            xyzs_tensor, [self.enc.plane_resolution, self.enc.plane_resolution*3], mode = "bilinear")

        # !Notice
        all_tensor = torch.cat([color_tensor, xyzs_tensor], dim = 1)# *2 - 1

        # triplane_feature0 = all_tensor
        # triplane_feature1 = all_tensor
        # return triplane_feature0, triplane_feature1
        return all_tensor

    def forward(self, data, ctx):
        # if self.geo_type == "flex":
        # use the number of data not batch size
        verts = self.renderer.flexicubes.verts.unsqueeze(0).repeat(len(data['verts']), 1, 1)
        tets = self.renderer.flexicubes.indices

        with torch.no_grad():
            tri_input = self.small_fwd(data)

        tri_output = self.unet(tri_input)

        assert torch.isnan(tri_output).sum() == 0, print("nan in triplane feature 2")
        dec_verts = self.decoder(tri_output, verts)

        pred_shape = self.sdf_mlp(dec_verts)
        assert torch.isnan(pred_shape).sum() == 0, print("nan in predicted shape primitives")

        pred_sdf, deformation = pred_shape[..., 0], pred_shape[..., 1:4]
        # if self.spob:
        pred_sdf = pred_sdf + self.radius - torch.sqrt((verts ** 2).sum(-1))

        for i in range(pred_sdf.shape[0]):
            if pred_sdf[i].min() >= 0 or pred_sdf[i].max() <= 0:
                print(f'004occur at object {data["uuid"][i]}!', pred_sdf[i].max(), pred_sdf[i].min())
                pred_sdf[i] -= (pred_sdf[i].min().detach() / 8 + pred_sdf[i].max().detach() * 7 / 8)

        weight = None
        # if self.geo_type == "flex":
        grid_feat = torch.index_select(input=dec_verts, index=self.renderer.flexicubes.indices.reshape(-1),dim=1)
        grid_feat = grid_feat.reshape(
            dec_verts.shape[0],self.renderer.flexicubes.indices.shape[0],
            self.renderer.flexicubes.indices.shape[1] * dec_verts.shape[-1]
            )
        weight = self.weight_mlp(grid_feat)
        weight = weight * 0.1

        result, verts, faces = self.renderer(data, pred_sdf, deformation, verts, tets, weight=weight)

        # sdf_pos = pred_sdf[pred_sdf>=0]
        # sdf_neg = pred_sdf[pred_sdf<0]
        # result["wzy_loss"] = (torch.abs(sdf_pos-1).sum() + torch.abs(sdf_neg+1).sum()) / (pred_sdf.shape[0]*pred_sdf.shape[1])

        resolution = data['resolution_img']
        color_gt = data['render_imgs']
        result['gt_render_imgs'] = data['render_imgs']

        img_list = []

        mvp_size = data['mvp'][0].shape[0]
        for vert, face in zip(verts, faces):
            mvp = data['mvp'][i]

            face = face.int()

            verts_clip = torch.bmm(
                F.pad(vert, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1),
                mvp.permute(0, 2, 1)).float()  # [B, N, 4]
            rast, rast_db = dr.rasterize(ctx, verts_clip, face, resolution)
            # print(verts.shape, faces.shape)
            alpha, _ = dr.interpolate(torch.ones_like(vert[:, :1]).unsqueeze(0), rast, face)  # [1, H, W, 1]
            xyzs, _ = dr.interpolate(vert, rast, face)  # [1, H, W, 3]
            # print(alpha.shape, xyzs.shape)
            xyzs = xyzs.view(-1, 3)
            mask = (alpha > 0).view(-1).detach()
            # do the lighting here since we have normal from mesh now.
            albedo = torch.zeros_like(xyzs, dtype=torch.float32)

            albedo = albedo.view(-1, resolution[0], resolution[1], 3)
            albedo[:mvp_size] = albedo[:mvp_size] - 1
            albedo = albedo.view(-1, 3)

            # print("4.0:{}".format(torch.cuda.memory_allocated(0)))

            # print("4.0:{}".format(torch.cuda.memory_allocated(0)))
            if mask.any():
                dec_d_pcd = self.decoder(tri_output[i].unsqueeze(0), xyzs[mask].unsqueeze(0))
                masked_albedo = self.rgb_mlp(dec_d_pcd)
                albedo[mask] = masked_albedo.squeeze(0).float()

                del masked_albedo
            albedo = albedo.view(-1, resolution[0], resolution[1], 3)
            albedo = (albedo * 0.5 + 0.5).clip(0, 1)

            img_list.append(albedo[:mvp_size])
            del albedo

        imgs = torch.stack(img_list, dim=0)
        result['albedo_list'] = imgs

        color_loss = self.mse_loss(imgs, color_gt)
        result.update({
            'color_gt': color_gt
        })

        result['color_loss'] = color_loss
        result["reg_loss"] = torch.norm(tri_input)

        result["tri_input"], result["tri_output"]= tri_input, tri_output

        return result, verts, faces

    def decode(self, data, tri_output):
        # if self.geo_type == "flex":
        tet_verts = self.renderer.flexicubes.verts.unsqueeze(0)
        tet_indices = self.renderer.flexicubes.indices

        dec_verts = self.decoder(tri_output, tet_verts)
        out = self.sdf_mlp(dec_verts)

        weight = None
        # if self.geo_type == "flex":
        grid_feat = torch.index_select(input=dec_verts, index=self.renderer.flexicubes.indices.reshape(-1),dim=1)
        grid_feat = grid_feat.reshape(
            dec_verts.shape[0], self.renderer.flexicubes.indices.shape[0], 
            self.renderer.flexicubes.indices.shape[1] * dec_verts.shape[-1]
            )
        weight = self.weight_mlp(grid_feat)
        weight = weight * 0.1

        pred_sdf, deformation = out[..., 0], out[..., 1:]
        # if self.spob:
        pred_sdf = pred_sdf + self.radius - torch.sqrt((tet_verts**2).sum(-1))

        _, verts, faces = self.renderer(data, pred_sdf, deformation, tet_verts, tet_indices, weight= weight)
        return verts[0].unsqueeze(0), faces[0].int()

    @torch.no_grad()
    def color_query(self, data, verts):
        tri_input = self.small_fwd(data)

        tri_output = self.unet(tri_input)

        dec_verts = self.decoder(tri_output, verts)
        colors = self.rgb_mlp(dec_verts)
        return colors

    @torch.no_grad()
    def render_color_img(self, verts, faces, data, ctx, mvp=None, idx=0):
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
            tri_input = self.small_fwd(data)
            tri_output = self.unet(tri_input)  # unet2 moved to trdgen main class
            masked_albedo = self.rgb_mlp(self.decoder(tri_output, xyzs[mask].unsqueeze(0))).squeeze(0)
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
        normal = result['normal'][0].detach().cpu().numpy()
        # normal_gt = result['normal_gt'][0].detach().cpu().numpy()
        alpha = (result['alpha'][0].detach().cpu().numpy() * 255).astype('uint8')
        alpha_gt = (result['alpha_gt'][0].detach().cpu().numpy() * 255).astype('uint8')

        # succ = True
        # try:
        depth = trans_depth(result['depth'])
        depth_gt = trans_depth(result['depth_gt'])
        # except:
        #     succ = False
        color_fine = color_fine[..., [2, 1, 0]].detach().cpu().numpy()
        color_fine = cv2.resize(color_fine[0], alpha.shape[1:3])[None, ...]

        color_fine_gt = result.get('color_gt')[0, ..., [2, 1, 0]].cpu().numpy()
        color_fine_gt = cv2.resize(color_fine_gt[0], alpha.shape[1:3])[None, ...]
        # if succ == True:
        img = np.concatenate((alpha, alpha_gt, depth, depth_gt), axis=2)
        # else:
        #     img = np.concatenate((alpha, alpha_gt, alpha, alpha_gt), axis=2)
        # if succ == True:
        cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}.png",
                    np.concatenate((normal[0], color_fine[0], color_fine_gt[0],
                                    cv2.applyColorMap(np.repeat(depth[0],3,axis=2).astype(np.uint8), cv2.COLORMAP_HOT)/255,
                                    cv2.applyColorMap(np.repeat(depth_gt[0],3,axis=2).astype(np.uint8), cv2.COLORMAP_HOT)/255),
                                    axis=1)*255)
        # else:
        #     cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}.png",
        # np.concatenate((normal[0], color_fine[0], color_fine_gt[0]), axis=1)*255)

        tri_input = result['tri_input'][0].mean(0).unsqueeze(-1)
        tri_input = (tri_input - tri_input.min()) / (tri_input.max() - tri_input.min())
        tri_input = tri_input.detach().cpu().numpy()
        tri_input = np.repeat(tri_input, 3, axis=2) * 255

        tri_output = result['tri_output'][0].mean(0).unsqueeze(-1)
        tri_output = (tri_output - tri_output.min()) / (tri_output.max() - tri_output.min())
        tri_output = tri_output.detach().cpu().numpy()
        tri_output = np.repeat(tri_output, 3, axis=2) * 255

        cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}-tri-mean.jpg",
                    cv2.applyColorMap(np.concatenate((tri_input, tri_output), axis=0).astype(np.uint8), cv2.COLORMAP_JET))

        self.ticker += 1

        i = self.ticker % result['tri_input'][0].shape[0]
        j = self.ticker % result['tri_output'][0].shape[0]

        tri_input = result['tri_input'][0][i].unsqueeze(-1)
        tri_input = (tri_input-tri_input.min())/(tri_input.max()-tri_input.min())
        tri_input = tri_input.detach().cpu().numpy()
        tri_input = np.repeat(tri_input, 3, axis=2) * 255

        tri_output = result['tri_output'][0][j].unsqueeze(-1)
        tri_output = (tri_output-tri_output.min())/(tri_output.max()-tri_output.min())
        tri_output = tri_output.detach().cpu().numpy()
        tri_output = np.repeat(tri_output, 3, axis=2) * 255

        cv2.imwrite(rdr_img_dir.as_posix() + f"/ep{epc}-chunk{chunk}-iter{iter_all}-tri{i}-{j}.jpg",
                    cv2.applyColorMap(np.concatenate((tri_input, tri_output), axis=0).astype(np.uint8), cv2.COLORMAP_JET))

        return img

    def export_mesh(self, data, out_path, tri_fea=None):
        '''Export mesh with vertice color for fast validation'''
        verts = data['verts']
        faces = data['faces']

        if tri_fea is None:
            colors = self.color_query(data, verts).squeeze().detach().cpu().numpy()
        else:
            dec_verts = self.decoder(tri_fea, verts.unsqueeze(0))
            colors = self.rgb_mlp(dec_verts).squeeze().detach().cpu().numpy()
        # Expect predicted colors value range from [-1, 1]
        colors = (colors * 0.5 + 0.5).clip(0, 1)

        verts = verts.squeeze().cpu().numpy()
        faces = faces[..., [2, 1, 0]].squeeze().cpu().numpy()

        # export the final mesh
        with torch.no_grad():
            # important, process=True leads to seg fault...
            mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors, process=False)
            mesh.export(out_path + '.ply')

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

        cv2.imwrite(f'{out_path}.png', img[..., [2, 1, 0]])
