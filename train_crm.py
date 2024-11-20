import os
import json
import argparse
import shutil
import kaolin
import re
import random

import numpy as np
import cv2
import trimesh
import torch
from pathlib import Path
import time
import nvdiffrast.torch as dr
from itertools import cycle
from tqdm import tqdm
import torch.distributed as dist

from model import TrDGen, MLPDecoder, PureDecoder, TriDecoder, FullDecoder, PCFullDecoder, PC2FullDecoder

from util.utils import lr_schedule, nan_to_num
from util.loss import chamfer_loss, laplacian_smooth_loss
from datasets.memory_dataset import MemoryDataset
from datasets.filesystem_dataset import FilesystemDataset
from datasets.fast_dataset import FastDataset
from metric.shape import vol_intersect_over_union, chamfer_distance, chamfer_distance_l1, silhouette_mask_iou

# from util.common import pcd_preprocess
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from util import utils
from util.utils import get_tri
from metric.lpips.lpips import LPIPS
import gc
def train(specs, config_path):
    input_specs = specs['Input']
    task_mode = input_specs['task_mode']
    scale = input_specs['scale']
    tet_grid_size = input_specs['tet_grid_size']
    dataset_type = input_specs.get('dataset_type', None)

    train_specs = specs['Train']
    num_epochs = train_specs['num_epochs']
    sup_mode = train_specs["mode"]
    sample_points_num = train_specs['sample_points_num']
    batch_size = train_specs['batch_size']
    l_rate = train_specs['learning_rate']['init']
    warmup_iter = train_specs['warm_up']
    scheduler_decay = train_specs['decay']
    eva_iter = train_specs['eva_iter']
    xyz_noise_scale = train_specs.get('xyz_noise_scale', 0.0)
    init_mdl_path = train_specs.get('init_mdl_path', None)
    eva_all_epoch = train_specs.get('eva_all_epoch')
    lambda_nc = train_specs.get('lambda_nc', 0)
    grad_acc = train_specs.get('grad_acc', 1)
    lambda_lp = train_specs.get('lambda_lp', 0)
    lambda_wzy = train_specs.get('lambda_wzy', 0)
    save_chunk = train_specs.get('save_chunk')

    output_dirs = specs['Output']
    exp_name = Path(output_dirs['exp_name'])

    geo_type = specs["Train"].get("geo_type", "dmtet")

    if not exp_name.exists():
        exp_name.mkdir(exist_ok=True)

    # ensure output in one folder ( higher than 60% )
    if args.resume:
        exp_dir = Path(args.resume)
    else:
        exp_dir = exp_name / (time.strftime('%m_%d-%H_%M_%S', time.localtime(time.time() // 5 * 5)) + args.note)
        exp_dir.mkdir(exist_ok=True)
        shutil.copy(config_path, exp_dir)
    #
    opt_rec_dir = exp_dir / "opt_rec"
    opt_rec_dir.mkdir(exist_ok=True)
    cur_rec_dir = exp_dir / "cur_rec"
    cur_rec_dir.mkdir(exist_ok=True)
    val_all_opt_rec_dir = exp_dir / "val_all_opt_rec"
    val_all_opt_rec_dir.mkdir(exist_ok=True)
    rdr_img_dir = exp_dir / 'rdr_img'
    rdr_img_dir.mkdir(exist_ok=True)

    if args.resume:
        print('load previous checkpoint')
        checkpoint_path = cur_rec_dir / 'previous_params.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            epc = checkpoint['epoch']
            chunk = checkpoint['chunk']
            chunk_num = checkpoint['chunk_num']
            # assert chunk_num == input_specs['chunk_num']
            # assert chunk == 0
            start = epc * chunk_num + chunk
            iter_all = checkpoint['itr_all']
            cham_dist_opt = checkpoint['cham_dist_opt']  # current best chamfer distance
            rgb_err_opt = checkpoint['rgb_err_opt']  # current best rgb error
            all_cham_dist_opt = checkpoint['all_cham_dist_opt']  # current best chamfer distance
            all_rgb_err_opt = checkpoint['all_rgb_err_opt']  # current best rgb error
        else:
            raise RuntimeError('No resume checkpoint found, train without --resume')
    else:
        cham_dist_opt = 9999  # current best chamfer distance
        rgb_err_opt = 9999  # current best rgb error
        all_cham_dist_opt = 9999  # current best chamfer distance
        all_rgb_err_opt = 9999  # current best rgb error

        start = 0
        chunk = 0
        iter_all = 0  # total number of passed training iterations

    accelerator = Accelerator()
    # kwargs = DDPK(find_unused_parameters=True)
    # accelerator = Accelerator(kwargs_handlers=[kwargs])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device

    # dataset
    glctx = dr.RasterizeCudaContext()

    dataset, dataset_val_all = [None] * 2

    if dataset_type == 'memory':
        dataset = MemoryDataset(input_specs, train_specs, glctx, tet_grid_size, device, task_mode=task_mode)

        chunk_num = 1

    elif dataset_type == 'filesystem':
        dataset = FastDataset(input_specs, train_specs, glctx, device, chunk, mode='all')
        # dataset_val_all = FilesystemDataset(input_specs, train_specs, glctx, device, mode='val')

        chunk_num = dataset.num_chunk

    dataset_val = MemoryDataset(input_specs, train_specs, glctx, tet_grid_size, device, task_mode=task_mode,
                                validate=True)
    dataset_vis = MemoryDataset(input_specs, train_specs, glctx, tet_grid_size, device, task_mode=task_mode,
                                   validate=True, visual=True)

    # dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
    #                                                      collate_fn=dataset_val.collate)
    #
    # for _ in dataloader_val:
    #     print('@')

    if accelerator.is_main_process:
        writer = SummaryWriter(str(exp_dir))
    else:
        writer = None

    # Encoder & Decoder & 3-layer MLP HEAD
    if specs['Model'] == 'MLP_only':
        model = MLPDecoder(specs, exp_dir).to(device)
    elif specs['Model'] == 'Pure':
        model = PureDecoder(specs, exp_dir).to(device)
    elif specs['Model'] == 'Tri':
        model = TriDecoder(specs, exp_dir).to(device)
    elif specs['Model'] == 'Full':
        model = FullDecoder(specs, exp_dir).to(device)
    elif specs['Model'] == 'PCFull':
        model = PCFullDecoder(specs, exp_dir).to(device)
    elif specs['Model'] == 'PC2Full':
        model = PC2FullDecoder(specs, exp_dir).to(device)
    else:
        model = TrDGen(specs, exp_dir).to(device)

    # model.load_state_dict(torch.load("/home/zhengyi/workspace/Text-to-3D/exp/10_31-17_26_34unet-pp/latest_params.pth", map_location = "cuda:"+str(accelerator.process_index)))
    if init_mdl_path != None:
        model.load_state_dict(torch.load(init_mdl_path, map_location = "cuda:"+str(accelerator.process_index))['model_state_dict'], strict=False)
    
    # dataset
    # glctx = dr.RasterizeCudaContext()

    # dataset, dataset_val_all = [None] * 2

    # if dataset_type == 'memory' or dataset_type is None:
    #     dataset = MemoryDataset(input_specs, train_specs, glctx, tet_grid_size, device, task_mode=task_mode)

    # elif dataset_type == 'filesystem':
    #     dataset = FilesystemDataset(input_specs, train_specs, glctx, device, mode='all')
    #     # dataset_val_all = FilesystemDataset(input_specs, train_specs, glctx, device, mode='val')

    #     chunk_num = dataset.num_chunk

    # dataset_val = MemoryDataset(input_specs, train_specs, glctx, tet_grid_size, device, task_mode=task_mode,
    #                             validate=True)
    # dataset_vis = MemoryDataset(input_specs, train_specs, glctx, tet_grid_size, device, task_mode=task_mode,
    #                                validate=True, visual=True)

    # dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
    #                                                      collate_fn=dataset_val.collate)
    #
    # for _ in dataloader_val:
    #     print('@')

    if accelerator.is_main_process:
        writer = SummaryWriter(str(exp_dir))
    else:
        writer = None



    optimizer = torch.optim.AdamW(model.get_params(l_rate))#(model.parameters(), lr=l_rate)
    # DJ optimization schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda x: lr_schedule(x, warmup_iter, scheduler_decay))

    for timestamp in sorted(os.listdir(exp_name), reverse=True):
        if re.fullmatch('\d{1,2}_\d{1,2}-\d{1,2}_\d{1,2}_\d{1,2}_tet90', timestamp) \
                and (Path(exp_name) / timestamp / 'val_all_opt_rec' / 'best_shape_params.pth').exists():
            params = Path(exp_name) / timestamp / 'val_all_opt_rec' / 'best_shape_params.pth'
            print('load previous 90 tetrahedra checkpoint')
            model.load_state_dict(torch.load(params))

            for name, para in model.named_parameters():
                if 'encoder' in name:
                    para.requires_grad = False

            input_specs['tet_grid_size'] = 128
            tet_grid_size = input_specs['tet_grid_size']

            train_specs['batch_size'] //= 2
            batch_size = train_specs['batch_size']

            break

    if args.resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    lpips = LPIPS(net_type="vgg", device="cuda:" + str(accelerator.process_index))

    model, optimizer, scheduler, lpips = accelerator.prepare(model, optimizer, scheduler, lpips)
    # lpips = lpips.to(device)
    
    # verts_gt_val = dataset_val.data['gt_verts']  # load ground truth mesh for evaluation
    # faces_gt_val = dataset_val.data['gt_faces']

    # pcd_tr = torch.Tensor(dataset.pcd_tr[None, ...]).cuda()
    # CYF comment
    # the following shift/scale operation muted
    # not seem necessary for ShapeNet pcd
    # pcd_tr -= np.mean(pcd_tr, axis=0)
    # bbox_length = np.sqrt(np.sum((np.max(pcd_tr, axis=0) - np.min(pcd_tr, axis=0))**2) )
    # pcd_tr /= bbox_length
    #
    # xyz_tr, gt_pt_tr = pcd_preprocess(pcd_tr)

    # DJ comment: pretrain SDF
    # params = 'exp/net_params.pth'
    # if not os.path.exists(params):
    #     self.pre_train_mesh(num_iterations=1000, gt_pc=sampled_pc)
    #     torch.save(self.state_dict(), params)
    # else:
    #     self.load_state_dict(torch.load(params))

    # its_per_epc = num_tr_pts // pt_b_size

    # distance between camera and origin
    # radius = specs['Train']['radius']
    # mvps = random_scene(radius, num_epochs * its_per_epc)

    # no need to pretrain any more
    pretrain = specs['Pretrain']
    if pretrain['mode'] is not None:
        # find the most recent state dict
        params = None
        for timestamp in sorted(os.listdir(exp_name), reverse=True):
            if re.fullmatch('\d{1,2}_\d{1,2}-\d{1,2}_\d{1,2}_\d{1,2}', timestamp) \
                    and (Path(exp_name) / timestamp / 'net_params.pth').exists():
                params = Path(exp_name) / timestamp / 'net_params.pth'
                break

        # !NOTICE Here!
        if params:
            # print(params)
            model.load_state_dict(torch.load(params))
        else:
            BVH = None
            if pretrain['mode'] == 'sdf_gt':
                # init sdf
                import cubvh
                BVH = cubvh.cuBVH(dataset.verts, dataset.faces)

            # CVX-H initialization
            # Note: only for debug mode now; should expand dataloader for batch mode
            pcd_num = dataset.data[0]['gt_pcd'].shape[0]
            smpl_pcd = dataset.data[0]['gt_pcd'][random.sample(range(pcd_num), k=sample_points_num), ...]
            model.pre_train_mesh(pretrain, smpl_pcd, writer, scale, BVH=BVH, device=device)
            # model.pre_train_mesh(pretrain, None, writer, scale, BVH=BVH, device=device)
            torch.save(model.state_dict(), exp_dir / 'net_params.pth')

    try:
        tets = np.load('tets/{}_tets.npz'.format(tet_grid_size))
    except:
        tets = np.load('tets/{}_tets.npz'.format(90))
    tet_verts = (torch.tensor(tets['vertices'], dtype=torch.float32,
                               device=device) * 2 - 1) * scale / 0.95  # covers [-scale/0.95, scale/0.95]
    tet_verts = tet_verts[None, ]
    tet_indices = torch.tensor(tets['indices'], dtype=torch.long, device=device)

    # # save groundtruth
    # if accelerator.is_main_process:
    #     print('Save groundtruth')
    #     data = dataset_val.data
    #     for i in tqdm(range(len(data))):
    #         mesh = trimesh.Trimesh(vertices=data[i]['gt_verts'], faces=data[i]['gt_faces'])
    #         mesh.export(os.path.join(opt_rec_dir, f'mesh_{i}_gt.ply'))
    #         mesh = trimesh.PointCloud(vertices=data[i]['gt_pcd'], colors=data[i]['gt_pcd_colors'])
    #         mesh.export(os.path.join(opt_rec_dir, f'pcd_{i}_gt.ply'))
    #
    #     # save val_all_groundtruth
    #     if dataset_val_all is not None:
    #         dataset_val_all.load_chunk()
    #         dataloader = torch.utils.data.DataLoader(dataset_val_all, batch_size=1,
    #                                                  collate_fn=dataset_val_all.collate)
    #         for i, data in enumerate(tqdm(dataloader)):
    #             mesh = trimesh.Trimesh(vertices=data['gt_verts'][0].cpu(), faces=data['gt_faces'][0].cpu())
    #             mesh.export(os.path.join(val_all_opt_rec_dir, f'mesh_{i}_gt.ply'))
    #             mesh = trimesh.PointCloud(vertices=data['gt_pcd'][0].cpu(), colors=data['gt_pcd_colors'][0].cpu())
    #             mesh.export(os.path.join(val_all_opt_rec_dir, f'pcd_{i}_gt.ply'))
    #
    # if 'RANK' in os.environ:
    #     dist.barrier()

    dataloader_vis = torch.utils.data.DataLoader(dataset_vis, batch_size=1,
                                                         collate_fn=dataset_vis.collate)
    dataloader_vis = cycle(dataloader_vis.__iter__())

    seed = int(accelerator.process_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if save_chunk is None:
        save_chunk = chunk_num

    for tmp in range(start, num_epochs * chunk_num + 1):
        epc = tmp // chunk_num
        chunk = tmp % chunk_num
        model.train()

        # if chunk not in [39, 81, 136]:
        #     continue

        # if dataset_type == 'filesystem' and tmp % (chunk_num * eva_all_epoch) == 0 and tmp > 0:
        #     # try:
        #     all_cham_dist_opt, all_rgb_err_opt = eval_and_save(
        #         accelerator, writer, model, tet_verts, tet_indices,
        #         sample_points_num, iter_all, dataset_val_all, device,
        #         val_all_opt_rec_dir, cur_rec_dir, all_cham_dist_opt, all_rgb_err_opt, glctx,
        #         type='_all')
        #     # except Exception as e:
        #     #     print(e)
        #     #     print(f'\033[91mtest: 004 error occur\033[0m')

        if (chunk % save_chunk == 0) and accelerator.is_main_process:
            torch.save({
                'epoch': epc,
                'chunk': chunk,
                'chunk_num': chunk_num,
                'itr_all': iter_all,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'cham_dist_opt': cham_dist_opt,
                'rgb_err_opt': rgb_err_opt,
                'all_cham_dist_opt': all_cham_dist_opt,
                'all_rgb_err_opt': all_rgb_err_opt,
            }, cur_rec_dir / 'previous_params.pth')

        # accelerator.wait_for_everyone()

        if dataset_type == 'filesystem':
            dataset.load_chunk()
            # skip empty chunk
            if len(dataset) == 0:
                continue

        try:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                     collate_fn=dataset.collate, shuffle=True)
            dataloader = accelerator.prepare(dataloader)
        except:
            continue

        # k = 0.1
        # alpha = train_specs['lambda_color'] * np.clip(k * (epc / num_epochs - 0.5) + 0.05, 0, 0.1)

        # pcd_tr = dataloader.pcd_tr
        for iter, data in enumerate(dataloader):
            gc.collect()
            model.train()
            
            # mvp = mvps[iter_all]
            # pts = pcd_tr[:, itr * pt_b_size: (itr + 1) * pt_b_size].cuda()
            #
            if accelerator.is_main_process:
                print(f"Training@Epoch %i, Chunk %i, Itr %i" % (epc, chunk, iter_all), flush=True)

            #     print(data['uuid'])
            #     iter_all += 1
            # continue
            # make data and tet sample size consistent
            tet_verts_batch = tet_verts.repeat(data['gt_pcd'].shape[0], 1, 1)
            result, verts, faces = model(data, tet_verts_batch, tet_indices, glctx, xyz_noise_scale=xyz_noise_scale)

            lap_loss_list, chamfer_loss_list = [], []
            # try:
            for i in range(len(verts)):
                lap_loss_list.append(laplacian_smooth_loss(verts[i], faces[i]))

                pred_points = kaolin.ops.mesh.sample_points(verts[i].unsqueeze(0), faces[i], sample_points_num)[0][0]
                chamfer_loss_list.append(chamfer_distance(pred_points.unsqueeze(0), data['gt_pcd'][i].unsqueeze(0)).mean())  # use dataloader; pcd_tr.cuda()

            result['lap_loss'] = sum(lap_loss_list) / len(lap_loss_list)
            result['chamfer_loss'] = sum(chamfer_loss_list) / len(chamfer_loss_list)

            # except Exception as e:
            #     print(e)
            #     print(f'\033[91mtrain: 004 error occur\033[0m')
            #     result['lap_loss'] = 0.
            #     result['chamfer_loss'] = 0.
            if train_specs['tex_sup_mode'] == 'blender':
                reso = result["albedo_list"].shape[-2]
                lpips_train = result["albedo_list"].view(-1, result["albedo_list"].shape[-3], result["albedo_list"].shape[-2], result["albedo_list"].shape[-1]).permute(0,3,1,2)
                lpips_gt = result["gt_render_imgs"].view(-1, result["gt_render_imgs"].shape[-3], result["gt_render_imgs"].shape[-2], result["gt_render_imgs"].shape[-1]).permute(0,3,1,2)

                if reso > 256:
                    startx = random.randint(0, reso-256-1)
                    starty = random.randint(0, reso-256-1)
                    lpips_train = lpips_train[:,:,startx:startx+256,starty:starty+256]
                    lpips_gt = lpips_gt[:,:,startx:startx+256,starty:starty+256]

                result['lpips_loss'] = lpips(lpips_train, lpips_gt)
            else:
                result['lpips_loss'] = 0
            #                                )
            if sup_mode == "rnd":
                # shape supervision I: rendered silhouette and depth losses
                loss = result['silhouette_loss']+ result['depth_loss'] + specs['Train']['lambda_smooth'] * result['lap_loss'] \
                    + train_specs['lambda_color'] * result['color_loss'] + train_specs['lambda_l2'] * result['reg_loss'] \
                    + lambda_nc * result['nc_loss'] + lambda_lp *result['lpips_loss'] + lambda_wzy * result["wzy_loss"]
                if geo_type == "flex":
                    loss = loss + result["flex_surf_loss"] * 0.5 * 0.0 + result["flex_weight_loss"] * 0.1 * 0.1
            elif sup_mode == "pcd":
                # shape supervision II: Chamfer distance loss
                loss = result['chamfer_loss'] + specs['Train']['lambda_smooth'] * result['lap_loss']
            else:
                raise ValueError("Error! Unrecognized supervision mode.")

            
            # loss.backward()
            # start = time.time()
            accelerator.backward(loss)
            # print(f'backward: {time.time() - start}')

            # # 打印梯度大小
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"iter {iter_all+1}, Layer {name}: Grad Max: {param.grad.data.max()}, Grad Min: {param.grad.data.min()}")

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).sum() > 0:
            #             print("nan", name, torch.isnan(param.grad).sum())
            #         param.grad[torch.isnan(param.grad)] = 0
            #         if torch.isinf(param.grad).sum() > 0:
            #             print("inf", name, torch.isinf(param.grad).sum())
            #         param.grad[torch.isinf(param.grad)] = 0

            params = [param for param in model.parameters() if param.grad is not None]
            if len(params) > 0:
                flat = torch.cat([param.grad.flatten() for param in params])
                if torch.isnan(flat).any():
                    print('==> find nan values')
                    print('==> nan grad')  # We should keep track of this for nan!!!!!!
                    for name, p in model.named_parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any():
                                print(name)
                nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                grads = flat.split([param.numel() for param in params])
                for param, grad in zip(params, grads):
                    param.grad = grad.reshape(param.shape)

            if iter_all % grad_acc == 0:
                optimizer.step()
                scheduler.step()  # DJ optimization schedule
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                
            # normal = result['normal_image'][0].detach().cpu().numpy()
            # cv2.imwrite(model.exp_dir.as_posix() + "/last_normal" +".png", normal*255)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                writer.add_scalar('train/silhouette', result['silhouette_loss'], iter_all)
                writer.add_scalar('train/depth', result['depth_loss'], iter_all)
                writer.add_scalar('train/Chamfer', result['chamfer_loss'], iter_all)
                writer.add_scalar('train/smooth', result['lap_loss'], iter_all)
                writer.add_scalar('train/color', result['color_loss'], iter_all)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_all)
                writer.add_scalar('train/lpips', result['lpips_loss'], iter_all)
                writer.add_scalar('train/wzy', result['wzy_loss'], iter_all)
                if geo_type == "flex":
                    writer.add_scalar('train/flex_weight', result['flex_weight_loss'], iter_all)
                    writer.add_scalar('train/flex_surf', result['flex_surf_loss'], iter_all)

            # perform evaluation every eva_iter iteration
            if iter_all % eva_iter == 0 or (iter_all < 500 and iter_all%20 ==0):
                if accelerator.is_main_process and train_specs['tex_sup_mode'] == 'blender':
                    train_img_dir = exp_dir / 'train_img'
                    train_img_dir.mkdir(exist_ok=True)
                    try:
                        albedos = result["albedo_list2"][0]
                    except:
                        albedos = result["albedo_list"][0]
                    albedo_tensor = torch.cat((albedos[0], albedos[1], albedos[2], albedos[3]), dim=1)
                    albedo_tensor = albedo_tensor[..., [2, 1, 0]].detach().cpu().numpy()
                    cv2.imwrite(exp_dir.as_posix() + f"/train_img/ep{epc}-chunk{chunk}-iter{iter_all}-train-"
                                                     f"{accelerator.process_index}-"
                                                     f"{data.get('prompt',[['null']])[0][0].replace(' ', '-')}.jpg",
                                albedo_tensor
                                * 255)
                # if accelerator.is_main_process:
                # print(accelerator.process_index, data['prompt'], data['uuid'])
                model.eval()
                # try:
                cham_dist_opt, rgb_err_opt = eval_and_save(accelerator, writer, model, tet_verts, tet_indices,
                                                            sample_points_num, iter_all, dataset_val, device,
                                                            opt_rec_dir, cur_rec_dir, cham_dist_opt, rgb_err_opt, glctx
                                                            )
                # except:
                #     print("eval failed, continue to train")

                # except Exception as e:
                #     print(e)
                #     print(f'\033[91mvalidation: 004 error occur\033[0m')
                # try:

                if accelerator.is_main_process:
                    # visualize
                    data_vis = next(dataloader_vis)
                    if task_mode != 'debug':
                        print(data_vis['uuid'],data_vis.get('prompt'))
                    # print(data_vis['triview_color'].shape)
                    # print(data_vis['triview_xyz'].shape)
                    # try:
                    tnew_list = [0,100,200,300,400,500,600,700,800,900,999]
                    with torch.no_grad():
                        result, verts, faces = accelerator.unwrap_model(model)(data_vis, tet_verts, tet_indices, glctx, tnew=tnew_list[(iter_all//eva_iter)%len(tnew_list)])

                    color_fine, _ = accelerator.unwrap_model(model).render_color_img(verts[0].unsqueeze(dim=0), faces[0],
                                                                            data_vis, glctx)
                    color_fine_gt, _ = accelerator.unwrap_model(model).render_color_img(verts[0].unsqueeze(dim=0), faces[0],
                                                                            data_vis, glctx, gt=True)
                    color_shape_gt, xyzs = accelerator.unwrap_model(model).render_color_img(data_vis['gt_verts'][0].unsqueeze(dim=0), data_vis['gt_faces'][0],
                                                                            data_vis, glctx, gt=True)
                    color_shape_gt2 = color_shape_gt[..., [2, 1, 0]].detach().cpu().numpy()
                    xyzs2 = xyzs[..., [2, 1, 0]].detach().cpu().numpy()
                    # cv2.imwrite(exp_dir.as_posix() + f"/ep{epc}-chunk{chunk}-rotate.jpg", np.concatenate((xyzs2[0], color_shape_gt2[0]), axis=1)*255)

                    # color_shape_gt, xyzs = accelerator.unwrap_model(model).render_color_img(data_vis['gt_verts'][0].unsqueeze(dim=0), data_vis['gt_faces'][0],
                    #                                                         data_vis, glctx, gt=True, mvp = side_scene_()[None,...].to(device))

                    # color_shape_gt[2] = color_shape_gt[2].flip(0)
                    # color_shape_gt[5] = color_shape_gt[5].flip(0)
                    # color_tensor1 = torch.cat((color_shape_gt[0], color_shape_gt[1], color_shape_gt[2]), dim=1)
                    # color_tensor2 = torch.cat((color_shape_gt[3], color_shape_gt[4], color_shape_gt[5]), dim=1)
                    # color_tensor = torch.cat((color_tensor1, color_tensor2.flip(0)), dim = 0)
                    # color_tensor = color_tensor[..., [2, 1, 0]].detach().cpu().numpy()

                    # xyzs[2] = xyzs[2].flip(0)
                    # xyzs[5] = xyzs[5].flip(0)
                    # xyzs_tensor1 = torch.cat((xyzs[0], xyzs[1], xyzs[2]), dim=1)
                    # xyzs_tensor2 = torch.cat((xyzs[3], xyzs[4], xyzs[5]), dim=1)
                    # xyzs_tensor = torch.cat((xyzs_tensor1, xyzs_tensor2.flip(0)), dim = 0)
                    # xyzs_tensor = xyzs_tensor.detach().cpu().numpy()

                    tri_img_dir = exp_dir / 'tri_img'
                    tri_img_dir.mkdir(exist_ok=True)
                    # cv2.imwrite(exp_dir.as_posix() + f"/tri_img/ep{epc}-chunk{chunk}-iter{iter_all}-side-color.jpg",color_tensor * 255)
                    # cv2.imwrite(exp_dir.as_posix() + f"/tri_img/ep{epc}-chunk{chunk}-iter{iter_all}-side-xyz.jpg",xyzs_tensor * 255) 
                    if specs["Train"].get("random_bg", False):
                        color = data_vis['triview_color'][0].permute(0,3,1,2).clone()
                        xyzs = data_vis['triview_xyz'][0].permute(0,3,1,2).clone()
                        crgb = random.random()
                        cxyz = random.random()
                        for j in range(6):
                            mask = torch.mean(xyzs[j], dim = 0, keepdim=False)
                            mask = (mask == 0)
                            for k in range(3):
                                color[j][k][mask] = crgb
                                xyzs[j][k][mask] = cxyz          
                                    
                        cv2.imwrite(exp_dir.as_posix() + f"/tri_img/ep{epc}-chunk{chunk}-iter{iter_all}-side-color-gt.jpg",get_tri(color, blender= True if specs["Train"].get("tex_sup_mode", "pcd-nnb") == 'blender' else False, c=crgb).permute(1,2,0)[..., [2, 1, 0]].detach().cpu().numpy() * 255)
                        cv2.imwrite(exp_dir.as_posix() + f"/tri_img/ep{epc}-chunk{chunk}-iter{iter_all}-side-xyz-gt.jpg",get_tri(xyzs, blender= True if specs["Train"].get("tex_sup_mode", "pcd-nnb") == 'blender' else False, c=cxyz).permute(1,2,0).detach().cpu().numpy() * 255)
                        
                    # color_shape_gt, xyzs = accelerator.unwrap_model(model).render_color_img(data_vis['gt_verts'][0].unsqueeze(dim=0), data_vis['gt_faces'][0],
                    #                                                         data_vis, glctx, mvp = sds_scene(6)[None,...].to(device))
                    # color_tensor1 = torch.cat((color_shape_gt[0], color_shape_gt[1], color_shape_gt[2]), dim=1)
                    # color_tensor2 = torch.cat((color_shape_gt[3], color_shape_gt[4], color_shape_gt[5]), dim=1)
                    # color_tensor = torch.cat((color_tensor1, color_tensor2), dim = 0)
                    # color_tensor = color_tensor[..., [2, 1, 0]].detach().cpu().numpy()

                    # xyzs_tensor1 = torch.cat((xyzs[0], xyzs[1], xyzs[2]), dim=1)
                    # xyzs_tensor2 = torch.cat((xyzs[3], xyzs[4], xyzs[5]), dim=1)
                    # xyzs_tensor = torch.cat((xyzs_tensor1, xyzs_tensor2), dim = 0)
                    # xyzs_tensor = xyzs_tensor.detach().cpu().numpy()


                    # cv2.imwrite(exp_dir.as_posix() + f"/tri_img/ep{epc}-chunk{chunk}-iter{iter_all}-sds-color.jpg",color_tensor * 255)
                    # cv2.imwrite(exp_dir.as_posix() + f"/tri_img/ep{epc}-chunk{chunk}-iter{iter_all}-sds-xyz.jpg",xyzs_tensor * 255)

                    img = accelerator.unwrap_model(model).visualize(result, epc, chunk, color_fine, color_fine_gt, rdr_img_dir, iter_all)
                    # only show the first validation mesh
                    writer.add_image('eval/visual', img[0], iter_all, dataformats='HWC')
                    # except Exception as e:
                    #     print(e)
                    #     print(f'\033[91mvisual: 004 error occur\033[0m')
                    # except:
                    #     print("vis failed")
            accelerator.wait_for_everyone()
            iter_all += 1

    return# model


def eval_and_save(accelerator, writer, model, tet_verts, tet_indices, sample_points_num, iter_all, dataset, device,
                  opt_rec_dir, cur_rec_dir, cham_dist_opt, rgb_err_opt, glctx, type=''):
    if isinstance(dataset, FilesystemDataset):
        num_chunk = dataset.num_chunk
    else:
        num_chunk = 1

    chamfer_dist_all, vol_IoU_all, lap_loss_all, verts_list_all, faces_list_all, gt_verts_list_all, gt_faces_list_all, \
        pc_list_all, pc_colors_list_all = [], [], [], [], [], [], [], [], []
    chamfer_l1_all, mask_iou_all, color_loss_all, depth_loss_all, silhouette_loss_all = [], [], [], [], []
    triview_color_list_all, triview_xyz_list_all = [], []

    for _ in range(num_chunk):
        if isinstance(dataset, FilesystemDataset):
            dataset.load_chunk()
            # skip empty chunk
            if len(dataset) == 0:
                continue

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                         collate_fn=dataset.collate)

        dataloader = accelerator.prepare(dataloader)

        # val_len = len(dataloader_val)
        for data_val in tqdm(dataloader):
            chamfer_dist, vol_IoU, lap_loss, verts_list, faces_list, gt_verts_list, gt_faces_list, pc_list, pc_colors_list = \
                [], [], [], [], [], [], [], [], []
            chamfer_l1, mask_iou, color_loss, depth_loss, silhouette_loss = [], [], [], [], []
            triview_color_list, triview_xyz_list = [], []
            with torch.no_grad():
                result, verts, faces = accelerator.unwrap_model(model)(data_val, tet_verts, tet_indices, glctx)

            torch.cuda.empty_cache()
            # result = dict((key, value.detach().cpu()) for key, value in result.items())
            # verts = tuple(ele.detach().cpu() for ele in verts)
            # faces = tuple(ele.detach().cpu() for ele in faces)
            # data_val = dict((key, [torch.tensor(ele).detach().cpu() for ele in value] if isinstance(value, list) else value.
            #                  detach().cpu()) for key, value in data_val.items())

            for i in range(len(verts)):
                pred_points = kaolin.ops.mesh.sample_points(verts[i].unsqueeze(0), faces[i], sample_points_num)[0][0]
                chamfer_dist.append(
                    chamfer_distance(
                        pred_points.unsqueeze(0), data_val['gt_pcd'][i].unsqueeze(0)).detach().cpu().numpy())  # use dataloader
                vol_IoU.append(vol_intersect_over_union(verts[i].unsqueeze(dim=0),
                                                        faces[i].type(torch.LongTensor),
                                                        data_val['gt_verts'][i].unsqueeze(dim=0),
                                                        data_val['gt_faces'][i], device=device).detach().cpu().numpy())
                lap_loss.append(laplacian_smooth_loss(verts[i], faces[i]).detach().cpu().numpy())
                chamfer_l1.append(
                    chamfer_distance_l1(pred_points, data_val['gt_pcd'][i], device=device).detach().cpu().numpy())
                mask_iou.append(silhouette_mask_iou(result['alpha'][i], result['alpha_gt'][i]).detach().cpu().numpy())

                color_loss.append(result['color_loss'].detach().cpu().numpy())
                depth_loss.append(result['depth_loss'].detach().cpu().numpy())
                silhouette_loss.append(result['silhouette_loss'].detach().cpu().numpy())
                # for multiprocess metric gather
                verts_list.append(verts[i].detach().cpu().numpy())
                faces_list.append(faces[i].detach().cpu().numpy())
                gt_verts_list.append(data_val['gt_verts'][i].cpu().numpy())
                gt_faces_list.append(data_val['gt_faces'][i].cpu().numpy())
                pc_list.append(data_val['gt_pcd'].cpu().numpy())
                pc_colors_list.append(data_val['gt_pcd_colors'].cpu().numpy())

                if data_val.get('triview_color') is None:
                    triview_color_list.append(np.nan)
                else:
                    triview_color_list.append(data_val['triview_color'].cpu().numpy())

                if data_val.get('triview_xyz') is None:
                    triview_xyz_list.append(np.nan)
                else:
                    triview_xyz_list.append(data_val['triview_xyz'].cpu().numpy())

            chamfer_dist = accelerator.gather_for_metrics(chamfer_dist)
            vol_IoU = accelerator.gather_for_metrics(vol_IoU)
            lap_loss = accelerator.gather_for_metrics(lap_loss)
            mask_iou = accelerator.gather_for_metrics(mask_iou)
            chamfer_l1 = accelerator.gather_for_metrics(chamfer_l1)
            color_loss = accelerator.gather_for_metrics(color_loss)
            depth_loss = accelerator.gather_for_metrics(depth_loss)
            silhouette_loss = accelerator.gather_for_metrics(silhouette_loss)

            pc_list = accelerator.gather_for_metrics(pc_list)
            pc_colors_list = accelerator.gather_for_metrics(pc_colors_list)
            verts_list = accelerator.gather_for_metrics(verts_list)
            faces_list = accelerator.gather_for_metrics(faces_list)
            gt_verts_list = accelerator.gather_for_metrics(gt_verts_list)
            gt_faces_list = accelerator.gather_for_metrics(gt_faces_list)

            triview_color_list = accelerator.gather_for_metrics(triview_color_list)
            triview_xyz_list = accelerator.gather_for_metrics(triview_xyz_list)

            torch.cuda.empty_cache()

            chamfer_dist_all.extend(chamfer_dist)
            vol_IoU_all.extend(vol_IoU)
            lap_loss_all.extend(lap_loss)
            mask_iou_all.extend(mask_iou)
            chamfer_l1_all.extend(chamfer_l1)
            color_loss_all.extend(color_loss)
            depth_loss_all.extend(depth_loss)
            silhouette_loss_all.extend(silhouette_loss)

            pc_list_all.extend(pc_list)
            pc_colors_list_all.extend(pc_colors_list)
            verts_list_all.extend(verts_list)
            faces_list_all.extend(faces_list)
            gt_verts_list_all.extend(gt_verts_list)
            gt_faces_list_all.extend(gt_faces_list)

            triview_color_list_all.extend(triview_color_list)
            triview_xyz_list_all.extend(triview_xyz_list)

    chamfer_dist = chamfer_dist_all
    vol_IoU = vol_IoU_all
    lap_loss = lap_loss_all
    mask_iou = mask_iou_all
    chamfer_l1 = chamfer_l1_all
    color_loss = color_loss_all
    depth_loss = depth_loss_all
    silhouette_loss = silhouette_loss_all

    pc_list = pc_list_all
    pc_colors_list = pc_colors_list_all
    verts_list = verts_list_all
    faces_list = faces_list_all
    gt_verts_list = gt_verts_list_all
    gt_faces_list = gt_faces_list_all

    triview_color_list = triview_color_list_all
    triview_xyz_list = triview_xyz_list_all

    metric = {
        'chamfer_dist': sum(chamfer_dist) / len(chamfer_dist),
        'vol_IoU': sum(vol_IoU) / len(vol_IoU),
        'lap_loss': sum(lap_loss) / len(lap_loss),
        'Mask_IoU': sum(mask_iou) / len(mask_iou),
        'Chamfer_L1': sum(chamfer_l1) / len(chamfer_l1),
        'color_loss': sum(color_loss) / len(color_loss),
        'depth_loss': sum(depth_loss) / len(depth_loss),
        'silhouette_loss': sum(silhouette_loss) / len(silhouette_loss)
    }

    if accelerator.is_main_process:
        writer.add_scalar(f'eval{type}/Chamfer', metric['chamfer_dist'], iter_all)
        writer.add_scalar(f'eval{type}/vol_IoU', metric['vol_IoU'], iter_all)
        writer.add_scalar(f'eval{type}/smooth', metric['lap_loss'], iter_all)
        writer.add_scalar(f'eval{type}/Chamfer-L1', metric['Chamfer_L1'], iter_all)
        writer.add_scalar(f'eval{type}/Mask-IoU', metric['Mask_IoU'], iter_all)
        writer.add_scalar(f'eval{type}/color', metric['color_loss'], iter_all)
        writer.add_scalar(f'eval{type}/depth', metric['depth_loss'], iter_all)
        writer.add_scalar(f'eval{type}/silhouette', metric['silhouette_loss'], iter_all)

        print('ITER %i: C2-Dist, %f (%f); RGB Err, %f (%f); C1-Dist, %f; Mask_IoU, %f; DL, %f; Silh, %f' %
              (iter_all, metric['chamfer_dist'], cham_dist_opt, metric['color_loss'], rgb_err_opt,
               metric['Chamfer_L1'], metric['Mask_IoU'], metric['depth_loss'], metric['silhouette_loss']), flush=True)

    if 'RANK' in os.environ:
        dist.barrier()

    # img = model.visualize(result, epc, chunk)
    # # only show the first validation mesh
    # writer.add_image('eval/visual', img[0], iter_all, dataformats='HWC')

    zip_data = [(pc, pc_colors, verts, faces, gt_verts, gt_faces, triview_color, triview_xyz)
                for pc, pc_colors, verts, faces, gt_verts, gt_faces, triview_color, triview_xyz in
                zip(pc_list, pc_colors_list, verts_list, faces_list, gt_verts_list, gt_faces_list, triview_color_list, triview_xyz_list)]

    gpu_idx = accelerator.local_process_index

    def save_export(mode):
        if accelerator.is_main_process:
            if (opt_rec_dir / f'best_{mode}_params.pth').exists():
                shutil.move(opt_rec_dir / f'best_{mode}_params.pth', opt_rec_dir / f'2nd_{mode}_params.pth')
            torch.save(accelerator.unwrap_model(model).state_dict(), opt_rec_dir / f'best_{mode}_params.pth')

        # export mesh ( only eval_all )
        if type != '_all' or specs['ArchSpecs']['pc_enc'] == 'mix':
            return

        with accelerator.split_between_processes(zip_data) as tmp_data:
            for j, ele in enumerate(tmp_data):
                pc, pc_colors, verts, faces, gt_verts, gt_faces, triview_color, triview_xyz = ele

                # export gt
                if not os.path.exists(os.path.join(opt_rec_dir, f'{gpu_idx}_{j}_mesh.ply')):
                    mesh = trimesh.Trimesh(vertices=gt_verts, faces=gt_faces)
                    mesh.export(os.path.join(opt_rec_dir, f'{gpu_idx}_{j}_mesh.ply'))
                if not os.path.exists(os.path.join(opt_rec_dir, f'{gpu_idx}_{j}_pcd.ply')):
                    pcd = trimesh.PointCloud(vertices=pc[0], colors=pc_colors[0])
                    pcd.export(os.path.join(opt_rec_dir, f'{gpu_idx}_{j}_pcd.ply'))

                # export pred
                data = {
                    'gt_verts': torch.tensor(gt_verts, device=device).unsqueeze(0),
                    'gt_pcd': torch.tensor(pc, device=device), 
                    'gt_pcd_color': torch.tensor(pc_colors, device=device), 
                    'verts': torch.tensor(verts, device=device).unsqueeze(0),
                    'faces': torch.tensor(faces, device=device).unsqueeze(0),
                    'triview_color': torch.tensor(triview_color, device=device),
                    'triview_xyz': torch.tensor(triview_xyz, device=device),
                }
                if accelerator.unwrap_model(model).exp_uv_mesh:
                    accelerator.unwrap_model(model).export_mesh_wt_uv(glctx, data, opt_rec_dir, 
                    f'{gpu_idx}_{j}_{mode}', device)
                else:
                    accelerator.unwrap_model(model).export_mesh(data, opt_rec_dir, 
                    f'{gpu_idx}_{j}_{mode}', device)

    # shpae model selection: save current best one
    if metric['chamfer_dist'] < cham_dist_opt:
        save_export('shape')
        cham_dist_opt = metric['chamfer_dist']

    # texture model selection: save current best one
    if metric['color_loss'] < rgb_err_opt:
        save_export('texture')
        rgb_err_opt = metric['color_loss']


    # # for debug purpose, also save current model/performance
    # torch.save(accelerator.unwrap_model(model).state_dict(), cur_rec_dir / 'cur_params.pth')
    # for i in range(len(verts_list)):
    #     accelerator.unwrap_model(model).export_mesh(pc_list[i], pc_colors_list[i],
    #                                                 verts_list[i][None,], faces_list[i], cur_rec_dir, i)
    # print out major learning/validation information

    return cham_dist_opt, rgb_err_opt


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        default="config/",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--config_name", "-c",
        default="specs_clear.json",
        help="config filename",
    )
    arg_parser.add_argument(
        "--note", "-n",
        default="",
        help="extra file name added to the output dir",
    )
    arg_parser.add_argument(
        "--resume",
        default=None,
        help="resume training from given checkpoint",
    )
    # arg_parser.add_argument(
    #     "--file", "-f",
    #     required=True,
    #     help="input point cloud filepath, in csv or ply format",
    # )
    # arg_parser.add_argument(
    #     "--outdir", "-o",
    #     required=True,
    #     help="output directory of reconstruction",
    # )
    #
    args = arg_parser.parse_args()
    if args.resume:
        config_path = os.path.join(args.resume, args.config_name)
    else:
        config_path = os.path.join(args.exp_dir, args.config_name)
    specs = json.load(open(config_path))
    # torch.cuda.set_device(1)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train(specs, config_path)
