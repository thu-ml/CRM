
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from util.utils import safe_normalize
from util.loss import normal_consistency
from util.flexicubes_geometry import FlexiCubesGeometry

class Renderer(nn.Module):
    def __init__(self, tet_grid_size, camera_angle_num, scale, geo_type):
        super().__init__()

        self.tet_grid_size = tet_grid_size
        self.camera_angle_num = camera_angle_num
        self.scale = scale
        self.geo_type = geo_type
        self.glctx = dr.RasterizeCudaContext()

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        if self.geo_type == "flex":
            self.flexicubes = FlexiCubesGeometry(grid_res = self.tet_grid_size)   

    def forward(self, data, sdf, deform, verts, tets, training=False, weight = None):
        mvp = data['mvp']
        campos = data['campos']
        h, w = data['resolution']
        persp = data['persp']

        results = {}

        deform = torch.tanh(deform) / self.tet_grid_size * self.scale / 0.95
        if self.geo_type == "flex":
            deform = deform *0.5

            v_deformed = verts + deform

            verts_list = []
            faces_list = []
            reg_list = []
            n_shape = verts.shape[0]
            for i in range(n_shape): 
                verts_i, faces_i, reg_i = self.flexicubes.get_mesh(v_deformed[i], sdf[i].squeeze(dim=-1),
                with_uv=False, indices=tets, weight_n=weight[i], is_training=training)

                verts_list.append(verts_i)
                faces_list.append(faces_i)
                reg_list.append(reg_i)       
            verts = verts_list
            faces = faces_list

            flexicubes_surface_reg = torch.cat(reg_list).mean()
            flexicubes_weight_reg = (weight ** 2).mean()
            results["flex_surf_loss"] = flexicubes_surface_reg
            results["flex_weight_loss"] = flexicubes_weight_reg

        alpha_list, alpha_ctn_list, depth_list, norm_list, nc_list = [], [], [], [], []
        for i in range(len(verts)):
            alpha, alpha_ctn, depth, normal, face_normals = \
                render_mesh(self.glctx, verts[i], faces[i], mvp[i], campos[i], persp, resolution=(h, w))

            alpha_list.append(alpha)
            alpha_ctn_list.append(alpha_ctn)
            depth_list.append(depth)
            norm_list.append(normal)
            nc_list.append(normal_consistency(face_normals, faces[i]))

        alpha = torch.stack(alpha_list, dim=0)
        alpha_ctn = torch.stack(alpha_ctn_list, dim=0)
        depth = torch.stack(depth_list, dim=0)
        normal = torch.stack(norm_list, dim=0)
        results['nc_loss'] = torch.stack(nc_list, dim=0).mean()

        results['alpha'] = alpha
        results['alpha_ctn'] = alpha_ctn
        results['depth'] = depth
        results['normal_image'] = (normal + 1) / 2

        # results['normal_loss'] = normal_consistency(face_normals, faces)
        try:# TODO remove these lines
            results['alpha_gt'] = data['alpha']
            results['alpha_ctn_gt'] = data['alpha_ctn']
            results['depth_gt'] = data['depth']
            # results['image'] = color
            # results['weights_sum'] = alpha.squeeze(-1)

            # # if self.opt.lambda_2d_normal_smooth > 0 or self.opt.lambda_normal > 0:
            # normal_image = dr.antialias((normal + 1) / 2, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
            # results['normal_image'] = normal_image
            results['normal_gt'] = (data['normal'] + 1) / 2
            # regularizations
            # if self.training:
            
            # print(results['alpha_ctn_gt'].shape, results['depth'].shape)
            results['silhouette_loss'] = self.mse_loss(results['alpha_ctn'], results['alpha_ctn_gt'])
            # results['depth_loss'] = self.l1_loss(results['depth'], results['depth_gt'])
            results['depth_loss'] = self.mse_loss(results['depth'] * results['alpha_ctn_gt'].clone().detach(), results['depth_gt'])
        except:
            pass

        return results, verts, faces

def render_mesh(ctx, verts, faces, mvp, view_pos, persp, resolution, normal_only=False):
                # spp=iter_spp, num_layers=self.FLAGS.layers, msaa=True, background=None)['shaded']:
                # get normals

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

    faces = faces.int()

    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = safe_normalize(face_normals)

    vn = torch.zeros_like(verts)
    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn,
                     torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

    # rasterization
    verts_clip = torch.bmm(
        F.pad(verts, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1),
        mvp.permute(0, 2, 1)).float()  # [B, N, 4]
    # assert verts_clip.shape[0] > 0 and verts_clip.shape[1] > 0, print(verts_clip.shape, verts.shape)
    rast, rast_db = dr.rasterize(ctx, verts_clip, faces, resolution)

    alpha = (rast[..., 3:] > 0).float()
    # CYF Note: enable gradient via antialiasing
    alpha_ctn = dr.antialias(alpha, rast, verts_clip, faces).clamp(0, 1)
    # rast[..., 3:][rast[..., 3:] > 0] /= rast[..., 3:][rast[..., 3:] > 0]
    # xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces) # [B, H, W, 3]
    normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
    normal = safe_normalize(normal)

    if normal_only:
        return normal, face_normals

    xyzw = verts_clip @ torch.inverse(persp.T)
    depth_ = (xyzw[..., 2] / xyzw[..., 3]).unsqueeze(-1)

    depth_, _ = dr.interpolate(depth_, rast, faces, rast_db=rast_db,
                               diff_attrs=None if rast_db is None else 'all')
    depth = dr.antialias(depth_, rast, verts_clip, faces)

    return alpha, alpha_ctn, -depth, normal, face_normals