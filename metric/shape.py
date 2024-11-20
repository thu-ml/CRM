from tqdm import tqdm
import numpy as np
import torch
import kaolin
import pytorch3d.loss
from scipy.spatial.distance import cdist


###############################################################################
# Reconstruction metrics placed in the following block
###############################################################################

def vol_intersect_over_union(verts_pred, faces_pred, verts_gt, faces_gt, device=torch.device('cuda')):
    axis = torch.linspace(-1., 1., steps=101)
    p_x, p_y, p_z = torch.meshgrid(axis, axis, axis, indexing='ij')
    points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
    points = points.view(1, -1, 3).to(device)

    # CYF note: it's weird that
    # 1) cuda face inputs become cpu variable internally
    # 2) verts is required dim-3 but faces is dim-2
    in_pred = kaolin.ops.mesh.check_sign(verts_pred, faces_pred.to(device), points)
    in_gt = kaolin.ops.mesh.check_sign(verts_gt, faces_gt.to(device), points)

    intersection = torch.logical_and(in_pred, in_gt).sum().float()
    union = torch.logical_or(in_pred, in_gt).sum().float()

    return intersection / union


def cdist_pt3d(pt_pr, pt_gt, norm):
    """
    CYF Note:
    utilize the unified pytorch3d method to compute Chamfer distances
    """

    assert norm in [1, 2]
    loss, _ = pytorch3d.loss.chamfer_distance(pt_pr, pt_gt, norm=norm, point_reduction="mean", single_directional=False)

    return loss


def chamfer_distance(ptr_pred, ptr_gt):
    return kaolin.metrics.pointcloud.chamfer_distance(ptr_pred, ptr_gt).mean()


def chamfer_distance_l1(ptr_pred, ptr_gt, device):
    ptr_pred = ptr_pred.cpu().numpy()
    ptr_gt = ptr_gt.cpu().numpy()

    mat = cdist(ptr_pred, ptr_gt, metric='cityblock')
    n0, n1 = mat.shape
    return torch.tensor(mat.min(axis=0).sum() / n1 + mat.min(axis=1).sum() / n0, dtype=torch.float32, device=device)


def silhouette_mask_iou(silhouette_pred, silhouette_gt):
    tmp = torch.stack([silhouette_pred, silhouette_gt], dim=0)
    inter = torch.all(tmp, dim=0).sum(dim=(1, 2, 3))
    union = torch.any(tmp, dim=0).sum(dim=(1, 2, 3))
    return (inter / union).mean()


###############################################################################
# Generative metrics placed in the following block
###############################################################################

"""
CYF Note:
The following methods are for evaluating generative metrics, i.e., MMD, COV, 1-NNA.
They are borrowed or adapted from two code bases, i.e., LION, and Diffusion-SDF
"""


def mmd_cov_1nna(ref_pcds, gen_pcds, b_size):
    results = {}

    # MMD and COV results
    M_rs_cd = _pairwise_cdist_(ref_pcds, gen_pcds, b_size)
    mmd_cd, cov_cd = lgan_mmd_cov(M_rs_cd.t())
    results["MMD-CD"] = mmd_cd
    results["COV-CD"] = cov_cd

    # 1-NN results
    M_rr_cd = _pairwise_cdist_(ref_pcds, ref_pcds, b_size)
    M_ss_cd = _pairwise_cdist_(gen_pcds, gen_pcds, b_size)
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results["1-NNA-CD"] = one_nn_cd_res['acc']

    return results


def _pairwise_cdist_(ref_pcds, gen_pcds, b_size):
    N_ref = ref_pcds.shape[0]
    N_gen = gen_pcds.shape[0]
    all_cd = []
    iterator = range(N_gen)

    with tqdm(iterator) as pbar:
        for gen_b_start in pbar:
            pbar.set_description("Files evaluated: {}/{}".format(gen_b_start, N_gen))
            gen_batch = gen_pcds[gen_b_start]

            cd_lst = []
            for ref_b_start in range(0, N_ref, b_size):
                ref_b_end = min(N_ref, ref_b_start + b_size)
                ref_batch = ref_pcds[ref_b_start: ref_b_end]

                b_size_ref = ref_batch.size(0)
                gen_batch_exp = gen_batch.view(1, -1, 3).expand(b_size_ref, -1, -1)
                gen_batch_exp = gen_batch_exp.contiguous()

                cd_b, _ = pytorch3d.loss.chamfer_distance(gen_batch_exp, ref_batch, norm=1, batch_reduction=None,
                                                          point_reduction="mean", single_directional=False)
                cd_lst.append(cd_b)

            cd_lst = torch.cat(cd_lst, dim=0)
            all_cd.append(cd_lst[:, None])

    all_cd = torch.cat(all_cd, dim=1)  # N_ref x N_gen

    return all_cd


def lgan_mmd_cov(all_dist):
    # all dist shape = [number of generated pcds, number of ref pcds]
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()

    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)

    return mmd, cov,


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)

    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })

    return s
