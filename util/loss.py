'''
Training loss
'''
import torch
import kaolin


def laplacian_uniform(verts, faces):
    V = verts.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].contiguous().flatten()
    jj = faces[:, [2, 0, 1]].contiguous().flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the correct diagonal
    L = torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    cor_idx = torch.stack((diag_idx, diag_idx), dim=0)
    cor_mat_ = torch.sparse_coo_tensor(cor_idx, adj_values, (V, V)).coalesce()
    cor_mat = torch.sparse_coo_tensor(cor_mat_.indices(), 1. / cor_mat_.values(), (V, V))

    return cor_mat.mm(L)


@torch.cuda.amp.autocast(enabled=False)
def laplacian_smooth_loss(verts, faces):
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss


def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge


@torch.cuda.amp.autocast(enabled=False)
def normal_consistency(face_normals, t_pos_idx):
    tris_per_edge = compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = 1.0 - term

    return torch.mean(torch.abs(term))


def chamfer_loss(verts, faces, pcd_tr, test_point_num=5000):
    pred_points = kaolin.ops.mesh.sample_points(verts.unsqueeze(0), faces, test_point_num)[0][0]
    loss = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), pcd_tr.cuda()).mean()
    return loss
