'''Main script training CRM'''
import os
import json
import argparse
import shutil
import random
from itertools import cycle
from pathlib import Path
import time

import kaolin
import cv2
import trimesh
import torch
import nvdiffrast.torch as dr
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import CRM
from util.utils import lr_schedule, nan_to_num
from util.loss import laplacian_smooth_loss
from datasets.memory_dataset import MemoryDataset
from datasets.fast_dataset import FastDataset
from metric.shape import vol_intersect_over_union, chamfer_distance, chamfer_distance_l1, silhouette_mask_iou
from metric.lpips.lpips import LPIPS


def train(args):
    if args.resume:
        config_path = os.path.join(args.resume, args.config_name)
    else:
        config_path = os.path.join(args.exp_dir, args.config_name)

    with open(config_path, encoding='utf8') as f:
        specs = json.load(f)

    input_specs = specs['Input']
    # task_mode = input_specs['task_mode']
    # tet_grid_size = input_specs['tet_grid_size']
    dataset_type = input_specs.get('dataset_type', None)

    train_specs = specs['Train']
    num_epochs = train_specs['num_epochs']
    # sup_mode = train_specs['mode']
    sample_points_num = train_specs['sample_points_num']
    batch_size = train_specs['batch_size']
    l_rate = train_specs['learning_rate']['init']
    warmup_iter = train_specs['warm_up']
    scheduler_decay = train_specs['decay']
    eva_iter = train_specs['eva_iter']
    # lambda_nc = train_specs.get('lambda_nc', 0)
    grad_acc = train_specs.get('grad_acc', 1)
    lambda_lp = train_specs.get('lambda_lp', 0)
    # lambda_wzy = train_specs.get('lambda_wzy', 0)
    save_chunk = train_specs.get('save_chunk')

    output_dirs = specs['Output']
    exp_name = Path(output_dirs['exp_name'])

    # geo_type = specs['Train'].get('geo_type', 'dmtet')

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
    opt_rec_dir = exp_dir / 'opt_rec'
    opt_rec_dir.mkdir(exist_ok=True)
    cur_rec_dir = exp_dir / 'cur_rec'
    cur_rec_dir.mkdir(exist_ok=True)
    val_all_opt_rec_dir = exp_dir / 'val_all_opt_rec'
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

    device = accelerator.device

    # dataset
    glctx = dr.RasterizeCudaContext()

    dataset = None

    if dataset_type == 'memory':
        dataset = MemoryDataset(input_specs, train_specs, glctx, device)
        chunk_num = 1

    elif dataset_type == 'filesystem':
        dataset = FastDataset(input_specs, train_specs, glctx, device, chunk, mode='all')
        chunk_num = dataset.num_chunk

    dataset_val = MemoryDataset(input_specs, train_specs, glctx, device, validate=True)
    dataset_vis = MemoryDataset(input_specs, train_specs, glctx, device, validate=True, visual=True)

    if accelerator.is_main_process:
        writer = SummaryWriter(str(exp_dir))
    else:
        writer = None

    # Encoder & Decoder & 3-layer MLP HEAD
    model = CRM(specs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate)#(model.parameters(), lr=l_rate)
    # optimization schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda x: lr_schedule(x, warmup_iter, scheduler_decay))

    if args.resume:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    lpips = LPIPS(net_type='vgg', device='cuda:' + str(accelerator.process_index))

    model, optimizer, scheduler, lpips = accelerator.prepare(model, optimizer, scheduler, lpips)

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

        if dataset_type == 'filesystem':
            dataset.load_chunk()
            # skip empty chunk
            if len(dataset) == 0:
                continue

        # try:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    collate_fn=dataset.collate, shuffle=True)
        dataloader = accelerator.prepare(dataloader)
        # except:
        #     continue

        for data in dataloader:
            model.train()
            # mvp = mvps[iter_all]
            # pts = pcd_tr[:, itr * pt_b_size: (itr + 1) * pt_b_size].cuda()
            #

            # mvp = mvps[iter_all]
            # pts = pcd_tr[:, itr * pt_b_size: (itr + 1) * pt_b_size].cuda()
            #
            if accelerator.is_main_process:
                print(f'Training@Epoch {epc}, Chunk {chunk}, Itr {iter_all}', flush=True)

            result, verts, faces = model(data, glctx)

            lap_loss_list, chamfer_loss_list = [], []

            for vert, face, gt_vert, gt_face in zip(verts, faces, data['verts'], data['faces']):
                lap_loss_list.append(laplacian_smooth_loss(vert, face))

                pred_points = kaolin.ops.mesh.sample_points(vert.unsqueeze(0), face, sample_points_num)[0][0]
                gt_points = kaolin.ops.mesh.sample_points(gt_vert.unsqueeze(0), gt_face, sample_points_num)[0][0]
                chamfer_loss_list.append(chamfer_distance(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).mean())

            result['lap_loss'] = sum(lap_loss_list) / len(lap_loss_list)
            result['chamfer_loss'] = sum(chamfer_loss_list) / len(chamfer_loss_list)

            # if train_specs['tex_sup_mode'] == 'blender':
            reso = result['albedo_list'].shape[-2]
            lpips_train = result['albedo_list'].view(
                -1, result['albedo_list'].shape[-3], result['albedo_list'].shape[-2], result['albedo_list'].shape[-1]
                ).permute(0,3,1,2)
            lpips_gt = result['gt_render_imgs'].view(
                -1, result['gt_render_imgs'].shape[-3], result['gt_render_imgs'].shape[-2], result['gt_render_imgs'].shape[-1]
                ).permute(0,3,1,2)

            if reso > 256:
                startx = random.randint(0, reso-256-1)
                starty = random.randint(0, reso-256-1)
                lpips_train = lpips_train[:,:,startx:startx+256,starty:starty+256]
                lpips_gt = lpips_gt[:,:,startx:startx+256,starty:starty+256]

            result['lpips_loss'] = lpips(lpips_train, lpips_gt)

            # if sup_mode == 'rnd':
            # shape supervision I: rendered silhouette and depth losses
            loss = result['silhouette_loss']+ result['depth_loss'] + specs['Train']['lambda_smooth'] * result['lap_loss'] \
                + train_specs['lambda_color'] * result['color_loss'] + lambda_lp *result['lpips_loss'] \
                + result['flex_weight_loss'] * 0.1 * 0.1

            accelerator.backward(loss)

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
                # writer.add_scalar('train/wzy', result['wzy_loss'], iter_all)
                # if geo_type == 'flex':
                writer.add_scalar('train/flex_weight', result['flex_weight_loss'], iter_all)
                writer.add_scalar('train/flex_surf', result['flex_surf_loss'], iter_all)

            # perform evaluation every eva_iter iteration
            if iter_all % eva_iter == 0 or (iter_all < 500 and iter_all % 20 ==0):
                if accelerator.is_main_process:
                    train_img_dir = exp_dir / 'train_img'
                    train_img_dir.mkdir(exist_ok=True)
                    # try:
                    #     albedos = result['albedo_list2'][0]
                    # except:
                    albedos = result['albedo_list'][0]
                    albedo_tensor = torch.cat((albedos[0], albedos[1], albedos[2], albedos[3]), dim=1)
                    albedo_tensor = albedo_tensor[..., [2, 1, 0]].detach().cpu().numpy()
                    cv2.imwrite(exp_dir.as_posix() + f'/train_img/ep{epc}-chunk{chunk}-iter{iter_all}-train-'
                                                     f'{accelerator.process_index}-'
                                                     f'{data.get("prompt",[["null"]])[0][0].replace(" ", "-")}.jpg',
                                albedo_tensor
                                * 255)

                model.eval()

                cham_dist_opt, rgb_err_opt = eval_and_save(accelerator, writer, model,
                                                            sample_points_num, iter_all, dataset_val, device,
                                                            opt_rec_dir, cham_dist_opt, rgb_err_opt, glctx
                                                            )

                if accelerator.is_main_process:
                    # visualize
                    data_vis = next(dataloader_vis)
                    # if task_mode != 'debug':
                    print(data_vis['uuid'],data_vis.get('prompt'))

                    # tnew_list = [0,100,200,300,400,500,600,700,800,900,999]
                    with torch.no_grad():
                        result, verts, faces = accelerator.unwrap_model(model)(data_vis, glctx)

                    color_fine, _ = accelerator.unwrap_model(model).render_color_img(verts[0].unsqueeze(dim=0), faces[0],
                                                                            data_vis, glctx)

                    img = accelerator.unwrap_model(model).visualize(result, epc, chunk, color_fine, rdr_img_dir, iter_all)
                    # only show the first validation mesh
                    writer.add_image('eval/visual', img[0], iter_all, dataformats='HWC')

            accelerator.wait_for_everyone()
            iter_all += 1

    return


def eval_and_save(accelerator, writer, model, sample_points_num, iter_all, dataset, device,
                  opt_rec_dir, cham_dist_opt, rgb_err_opt, glctx):
    if isinstance(dataset, FastDataset):
        num_chunk = dataset.num_chunk
    else:
        num_chunk = 1

    chamfer_dist_all, vol_iou_all, lap_loss_all, verts_list_all, faces_list_all = [], [], [], [], []
    gt_verts_list_all, gt_faces_list_all= [], []
    chamfer_l1_all, mask_iou_all, color_loss_all, depth_loss_all, silhouette_loss_all = [], [], [], [], []
    triview_color_list_all, triview_xyz_list_all = [], []

    for _ in range(num_chunk):
        if isinstance(dataset, FastDataset):
            dataset.load_chunk()
            # skip empty chunk
            if len(dataset) == 0:
                continue

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                         collate_fn=dataset.collate)

        dataloader = accelerator.prepare(dataloader)

        for data_val in tqdm(dataloader):
            chamfer_dist, vol_iou, lap_loss, verts_list, faces_list, gt_verts_list, gt_faces_list = [], [], [], [], [], [], []
            chamfer_l1, mask_iou, color_loss, depth_loss, silhouette_loss = [], [], [], [], []
            triview_color_list, triview_xyz_list = [], []
            with torch.no_grad():
                result, verts, faces = accelerator.unwrap_model(model)(data_val, glctx)

            torch.cuda.empty_cache()

            for vert, face, alpha, gt_vert, gt_face, gt_alpha in zip(
                verts, faces, result['alpha'], data_val['verts'], data_val['faces'], result['alpha_gt']
                ):
                pred_points = kaolin.ops.mesh.sample_points(vert.unsqueeze(0), face, sample_points_num)[0][0]
                gt_points = kaolin.ops.mesh.sample_points(gt_vert.unsqueeze(0), gt_face, sample_points_num)[0][0]
                chamfer_dist.append(
                    chamfer_distance(
                        pred_points.unsqueeze(0), gt_points.unsqueeze(0)).detach().cpu().numpy())  # use dataloader
                vol_iou.append(vol_intersect_over_union(vert.unsqueeze(dim=0),
                                                        face.type(torch.LongTensor),
                                                        gt_vert.unsqueeze(dim=0),
                                                        gt_face, device=device).detach().cpu().numpy())
                lap_loss.append(laplacian_smooth_loss(vert, face).detach().cpu().numpy())
                chamfer_l1.append(
                    chamfer_distance_l1(pred_points, gt_points, device=device).detach().cpu().numpy())
                mask_iou.append(silhouette_mask_iou(alpha, gt_alpha).detach().cpu().numpy())

                color_loss.append(result['color_loss'].detach().cpu().numpy())
                depth_loss.append(result['depth_loss'].detach().cpu().numpy())
                silhouette_loss.append(result['silhouette_loss'].detach().cpu().numpy())
                verts_list.append(vert.detach().cpu().numpy())
                faces_list.append(face.detach().cpu().numpy())
                gt_verts_list.append(gt_vert.cpu().numpy())
                gt_faces_list.append(gt_face.cpu().numpy())

                # if data_val.get('triview_color') is None:
                #     triview_color_list.append(np.nan)
                # else:
                triview_color_list.append(data_val['triview_color'].cpu().numpy())

                # if data_val.get('triview_xyz') is None:
                #     triview_xyz_list.append(np.nan)
                # else:
                triview_xyz_list.append(data_val['triview_xyz'].cpu().numpy())

            chamfer_dist = accelerator.gather_for_metrics(chamfer_dist)
            vol_iou = accelerator.gather_for_metrics(vol_iou)
            lap_loss = accelerator.gather_for_metrics(lap_loss)
            mask_iou = accelerator.gather_for_metrics(mask_iou)
            chamfer_l1 = accelerator.gather_for_metrics(chamfer_l1)
            color_loss = accelerator.gather_for_metrics(color_loss)
            depth_loss = accelerator.gather_for_metrics(depth_loss)
            silhouette_loss = accelerator.gather_for_metrics(silhouette_loss)

            verts_list = accelerator.gather_for_metrics(verts_list)
            faces_list = accelerator.gather_for_metrics(faces_list)
            gt_verts_list = accelerator.gather_for_metrics(gt_verts_list)
            gt_faces_list = accelerator.gather_for_metrics(gt_faces_list)

            triview_color_list = accelerator.gather_for_metrics(triview_color_list)
            triview_xyz_list = accelerator.gather_for_metrics(triview_xyz_list)

            torch.cuda.empty_cache()

            chamfer_dist_all.extend(chamfer_dist)
            vol_iou_all.extend(vol_iou)
            lap_loss_all.extend(lap_loss)
            mask_iou_all.extend(mask_iou)
            chamfer_l1_all.extend(chamfer_l1)
            color_loss_all.extend(color_loss)
            depth_loss_all.extend(depth_loss)
            silhouette_loss_all.extend(silhouette_loss)

            verts_list_all.extend(verts_list)
            faces_list_all.extend(faces_list)
            gt_verts_list_all.extend(gt_verts_list)
            gt_faces_list_all.extend(gt_faces_list)

            triview_color_list_all.extend(triview_color_list)
            triview_xyz_list_all.extend(triview_xyz_list)

    chamfer_dist = chamfer_dist_all
    vol_iou = vol_iou_all
    lap_loss = lap_loss_all
    mask_iou = mask_iou_all
    chamfer_l1 = chamfer_l1_all
    color_loss = color_loss_all
    depth_loss = depth_loss_all
    silhouette_loss = silhouette_loss_all

    verts_list = verts_list_all
    faces_list = faces_list_all
    gt_verts_list = gt_verts_list_all
    gt_faces_list = gt_faces_list_all

    triview_color_list = triview_color_list_all
    triview_xyz_list = triview_xyz_list_all

    metric = {
        'chamfer_dist': sum(chamfer_dist) / len(chamfer_dist),
        'vol_iou': sum(vol_iou) / len(vol_iou),
        'lap_loss': sum(lap_loss) / len(lap_loss),
        'Mask_IoU': sum(mask_iou) / len(mask_iou),
        'Chamfer_L1': sum(chamfer_l1) / len(chamfer_l1),
        'color_loss': sum(color_loss) / len(color_loss),
        'depth_loss': sum(depth_loss) / len(depth_loss),
        'silhouette_loss': sum(silhouette_loss) / len(silhouette_loss)
    }

    if accelerator.is_main_process:
        writer.add_scalar('eval/Chamfer', metric['chamfer_dist'], iter_all)
        writer.add_scalar('eval/vol_iou', metric['vol_iou'], iter_all)
        writer.add_scalar('eval/smooth', metric['lap_loss'], iter_all)
        writer.add_scalar('eval/Chamfer-L1', metric['Chamfer_L1'], iter_all)
        writer.add_scalar('eval/Mask-IoU', metric['Mask_IoU'], iter_all)
        writer.add_scalar('eval/color', metric['color_loss'], iter_all)
        writer.add_scalar('eval/depth', metric['depth_loss'], iter_all)
        writer.add_scalar('eval/silhouette', metric['silhouette_loss'], iter_all)

        print(f'ITER {iter_all}: C2-Dist, {metric["chamfer_dist"]:f} ({cham_dist_opt:f}); '
              f'RGB Err, {rgb_err_opt:f} ({rgb_err_opt:f}); C1-Dist, {metric["Chamfer_L1"]:f}; '
              f'Mask_IoU, {metric["Mask_IoU"]:f}; DL, {metric["depth_loss"]:f}; '
              f'Silh, {metric["silhouette_loss"]:f}' , flush=True)

    accelerator.wait_for_everyone()

    zip_data = [(verts, faces, gt_verts, gt_faces, triview_color, triview_xyz)
                for verts, faces, gt_verts, gt_faces, triview_color, triview_xyz in
                zip(verts_list, faces_list, gt_verts_list, gt_faces_list, triview_color_list, triview_xyz_list)]

    gpu_idx = accelerator.local_process_index

    def save_export(mode):
        if accelerator.is_main_process:
            if (opt_rec_dir / f'best_{mode}_params.pth').exists():
                shutil.move(opt_rec_dir / f'best_{mode}_params.pth', opt_rec_dir / f'2nd_{mode}_params.pth')
            torch.save(accelerator.unwrap_model(model).state_dict(), opt_rec_dir / f'best_{mode}_params.pth')

        # # export mesh ( only eval_all )
        # if type != '_all' or specs['ArchSpecs']['pc_enc'] == 'mix':
        #     return

        with accelerator.split_between_processes(zip_data) as tmp_data:
            for j, ele in enumerate(tmp_data):
                verts, faces, gt_verts, gt_faces, triview_color, triview_xyz = ele

                # export gt
                if not os.path.exists(os.path.join(opt_rec_dir, f'{gpu_idx}_{j}_mesh.ply')):
                    mesh = trimesh.Trimesh(vertices=gt_verts, faces=gt_faces)
                    mesh.export(os.path.join(opt_rec_dir, f'{gpu_idx}_{j}_mesh.ply'))

                # export pred
                data = {
                    'gt_verts': torch.tensor(gt_verts, device=device).unsqueeze(0),
                    'verts': torch.tensor(verts, device=device).unsqueeze(0),
                    'faces': torch.tensor(faces, device=device).unsqueeze(0),
                    'triview_color': torch.tensor(triview_color, device=device),
                    'triview_xyz': torch.tensor(triview_xyz, device=device),
                }

                accelerator.unwrap_model(model).export_mesh(data, f'{opt_rec_dir}/{gpu_idx}_{j}_{mode}')

    # shpae model selection: save current best one
    if metric['chamfer_dist'] < cham_dist_opt:
        save_export('shape')
        cham_dist_opt = metric['chamfer_dist']

    # texture model selection: save current best one
    if metric['color_loss'] < rgb_err_opt:
        save_export('texture')
        rgb_err_opt = metric['color_loss']

    return cham_dist_opt, rgb_err_opt


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--exp_dir', '-e',
        default='configs/',
        help='This directory should include experiment specifications in "specs.json"',
    )
    arg_parser.add_argument(
        '--config_name', '-c',
        default='specs_clear.json',
        help='config filename',
    )
    arg_parser.add_argument(
        '--note', '-n',
        default='',
        help='extra file name added to the output dir',
    )
    arg_parser.add_argument(
        '--resume',
        default=None,
        help='resume training from given checkpoint',
    )
    arguments = arg_parser.parse_args()
    # torch.cuda.set_device(1)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train(arguments)
