'''
Reconstruction metrics
'''
import torch
import kaolin
from scipy.spatial.distance import cdist


def vol_intersect_over_union(verts_pred, faces_pred, verts_gt, faces_gt, device=torch.device('cuda')):
    '''volume IOU calculation'''
    axis = torch.linspace(-1., 1., steps=101)
    p_x, p_y, p_z = torch.meshgrid(axis, axis, axis, indexing='ij')
    points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
    points = points.view(1, -1, 3).to(device)

    in_pred = kaolin.ops.mesh.check_sign(verts_pred, faces_pred.to(device), points)
    in_gt = kaolin.ops.mesh.check_sign(verts_gt, faces_gt.to(device), points)

    intersection = torch.logical_and(in_pred, in_gt).sum().float()
    union = torch.logical_or(in_pred, in_gt).sum().float()

    return intersection / union

def chamfer_distance(ptr_pred, ptr_gt):
    '''Chamfer distance L2 calculation'''
    return kaolin.metrics.pointcloud.chamfer_distance(ptr_pred, ptr_gt).mean()

def chamfer_distance_l1(ptr_pred, ptr_gt, device):
    '''Chamfer distance L1 calculation'''
    ptr_pred = ptr_pred.cpu().numpy()
    ptr_gt = ptr_gt.cpu().numpy()

    mat = cdist(ptr_pred, ptr_gt, metric='cityblock')
    n0, n1 = mat.shape
    return torch.tensor(mat.min(axis=0).sum() / n1 + mat.min(axis=1).sum() / n0, dtype=torch.float32, device=device)

def silhouette_mask_iou(silhouette_pred, silhouette_gt):
    '''Mask IOU calculation'''
    tmp = torch.stack([silhouette_pred, silhouette_gt], dim=0)
    inter = torch.all(tmp, dim=0).sum(dim=(1, 2, 3))
    union = torch.any(tmp, dim=0).sum(dim=(1, 2, 3))
    return (inter / union).mean()
