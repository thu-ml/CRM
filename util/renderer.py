'''
Transfer sdf -> mesh -> alpha, depth, color_image, normal
'''
import torch
from torch import nn
from torch.nn import functional as F
import nvdiffrast.torch as dr

from util.utils import safe_normalize
from util.flexicubes_geometry import FlexiCubesGeometry

class Renderer(nn.Module):
    '''Render class'''
    def __init__(self, tet_grid_size, camera_angle_num, scale):
        super().__init__()

        self.tet_grid_size = tet_grid_size
        self.camera_angle_num = camera_angle_num
        self.scale = scale
        self.glctx = dr.RasterizeCudaContext()

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.flexicubes = FlexiCubesGeometry(grid_res = self.tet_grid_size)

    def forward(self, data, sdf, deform, verts, tets, training=False, weight = None):
        '''Render main function'''
        mvps = data['mvp']
        # campos = data['campos']
        h, w = data['resolution']
        persp = data['persp']

        results = {}
        faces = None
        deform = torch.tanh(deform) / self.tet_grid_size
        deform = deform * 0.5

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

        alpha_list, depth_list, norm_list = [], [], []
        for vert, face, mvp in zip(verts, faces, mvps):
            alpha, depth, normal = render_mesh(self.glctx, vert, face, mvp, persp, resolution=(h, w))

            alpha_list.append(alpha)
            depth_list.append(depth)
            norm_list.append(normal)

        alpha = torch.stack(alpha_list, dim=0)
        depth = torch.stack(depth_list, dim=0)
        normal = torch.stack(norm_list, dim=0)

        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2

        results['alpha_gt'] = data['alpha']
        results['depth_gt'] = data['depth']
        results['normal_gt'] = (data['normal'] + 1) / 2

        results['silhouette_loss'] = self.mse_loss(results['alpha'], results['alpha_gt'])
        results['depth_loss'] = self.mse_loss(results['depth'] * results['alpha_gt'].clone().detach(), results['depth_gt'])

        return results, verts, faces

def render_mesh(ctx, verts, faces, mvp, persp, resolution):
    '''Rasterize single mesh and interpolate to get depth, alpha, normal'''
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
    rast, rast_db = dr.rasterize(ctx, verts_clip, faces, resolution)

    alpha = (rast[..., 3:] > 0).float()
    # CYF Note: enable gradient via antialiasing
    alpha = dr.antialias(alpha, rast, verts_clip, faces).clamp(0, 1)
    normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
    normal = safe_normalize(normal)

    xyzw = verts_clip @ torch.inverse(persp.T)
    depth_ = (xyzw[..., 2] / xyzw[..., 3]).unsqueeze(-1)

    depth_, _ = dr.interpolate(depth_, rast, faces, rast_db=rast_db,
                               diff_attrs=None if rast_db is None else 'all')
    depth = dr.antialias(depth_, rast, verts_clip, faces)

    return alpha, -depth, normal
