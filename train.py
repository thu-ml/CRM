"""
training script for imagedream
- the config system is similar with stable diffusion ldm code base(using omigaconf, yaml; target, params initialization, etc.)
- the training code base is similar with unidiffuser training code base using accelerate

"""

from omegaconf import OmegaConf
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import os.path as osp
import numpy as np
import os
import torch
from PIL import Image
import numpy as np
import wandb
from libs.base_utils import get_data_generator, PrintContext
from libs.base_utils import (
    setup,
    instantiate_from_config,
    dct2str,
    add_prefix,
    get_obj_from_str,
)
from absl import logging
from einops import rearrange
from imagedream.camera_utils import get_camera
from libs.sample import ImageDreamDiffusion
from rich import print


def train(config, unk):
    # using pipeline to extract models
    accelerator, device = setup(config, unk)
    with PrintContext(f"{'access STAT':-^50}", accelerator.is_main_process):
        print(accelerator.state)
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "no": torch.float32,
        "bf16": torch.bfloat16,
    }[accelerator.state.mixed_precision]

    num_frames = config.num_frames

    ################## load models ##################
    model_config = config.models.config
    model_config = OmegaConf.load(model_config)
    model = instantiate_from_config(model_config.model)
    state_dict = torch.load(config.models.resume, map_location="cpu")

    print(model.load_state_dict(state_dict, strict=False))
    print("loaded model from {}".format(config.models.resume))

    latest_step = 0
    if config.get("resume", False):
        print("resuming from specified workdir")
        ckpts = os.listdir(config.ckpt_root)
        if len(ckpts) == 0:
            print("no ckpt found")
        else:
            latest_ckpt = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]
            latest_step = int(latest_ckpt.split("-")[-1])
            print("loadding ckpt from ", osp.join(config.ckpt_root, latest_ckpt))
            unet_state_dict = torch.load(
                osp.join(config.ckpt_root, latest_ckpt), map_location="cpu"
            )
            print(model.model.load_state_dict(unet_state_dict, strict=False))

    elif config.models.get("resume_unet", None) is not None:
        unet_state_dict = torch.load(config.models.resume_unet, map_location="cpu")
        print(model.model.load_state_dict(unet_state_dict, strict=False))
        print(f"______ load unet from {config.models.resume_unet} ______")
    model.to(device)
    model.device = device
    model.clip_model.device = device

    ################# setup optimizer #################
    from torch.optim import AdamW
    from accelerate.utils import DummyOptim

    optimizer_cls = (
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.model.parameters(), **config.optimizer)

    ################# prepare datasets #################
    dataset = instantiate_from_config(config.train_data)
    eval_dataset = instantiate_from_config(config.eval_data)
    in_the_wild_images = (
        instantiate_from_config(config.in_the_wild_images)
        if config.get("in_the_wild_images", None) is not None
        else None
    )

    dl_config = config.dataloader
    dataloader = DataLoader(dataset, **dl_config, batch_size=config.batch_size)

    (
        model,
        optimizer,
        dataloader,
    ) = accelerator.prepare(model, optimizer, dataloader)

    generator = get_data_generator(dataloader, accelerator.is_main_process, "train")
    if config.get("sampler", None) is not None:
        sampler_cls = get_obj_from_str(config.sampler.target)
        sampler = sampler_cls(model, device, dtype, **config.sampler.params)
    else:
        sampler = ImageDreamDiffusion(
            model,
            mode=config.mode,
            num_frames=num_frames,
            device=device,
            dtype=dtype,
            camera_views=dataset.camera_views,
            offset_noise=config.get("offset_noise", False),
            ref_position=dataset.ref_position,
            random_background=dataset.random_background,
            resize_rate=dataset.resize_rate,
        )

    ################# evaluation code #################
    def evaluation():
        return_ls = []
        for i in range(
            accelerator.process_index, len(eval_dataset), accelerator.num_processes
        ):
            cond = eval_dataset[i]["cond"]

            images = sampler.diffuse("3D assets.", cond, n_test=2)
            images = np.concatenate(images, 0)
            images = [Image.fromarray(images)]
            return_ls.append(dict(images=images, ident=eval_dataset[i]["ident"]))
        return return_ls

    def evaluation2():
        # eval for common used in the wild image
        return_ls = []
        in_the_wild_images.init_item()
        for i in range(
            accelerator.process_index,
            len(in_the_wild_images),
            accelerator.num_processes,
        ):
            cond = in_the_wild_images[i]["cond"]
            images = sampler.diffuse("3D assets.", cond, n_test=2)
            images = np.concatenate(images, 0)
            images = [Image.fromarray(images)]
            return_ls.append(dict(images=images, ident=in_the_wild_images[i]["ident"]))
        return return_ls

    if latest_step == 0:
        global_step = 0
        total_step = 0
        log_step = 0
        eval_step = 0
        save_step = 0
    else:
        global_step = latest_step // config.total_batch_size
        total_step = latest_step
        log_step = latest_step + config.log_interval
        eval_step = latest_step + config.eval_interval
        save_step = latest_step + config.save_interval

    unet = model.model
    while True:
        item = next(generator)
        unet.train()
        bs = item["clip_cond"].shape[0]
        BS = bs * num_frames
        item["clip_cond"] = item["clip_cond"].to(device).to(dtype)
        item["vae_cond"] = item["vae_cond"].to(device).to(dtype)
        camera_input = item["cameras"].to(device)
        camera_input = camera_input.reshape((BS, camera_input.shape[-1]))

        gd_type = config.get("gd_type", "pixel")
        if gd_type == "pixel":
            item["target_images_vae"] = item["target_images_vae"].to(device).to(dtype)
            gd = item["target_images_vae"]
        elif gd_type == "xyz":
            item["target_images_xyz_vae"] = (
                item["target_images_xyz_vae"].to(device).to(dtype)
            )
            gd = item["target_images_xyz_vae"]
        elif gd_type == "fusechannel":
            item["target_images_vae"] = item["target_images_vae"].to(device).to(dtype)
            item["target_images_xyz_vae"] = (
                item["target_images_xyz_vae"].to(device).to(dtype)
            )
            gd = torch.cat(
                (item["target_images_vae"], item["target_images_xyz_vae"]), dim=0
            )
        else:
            raise NotImplementedError

        with torch.no_grad(), accelerator.autocast("cuda"):
            ip_embed = model.clip_model.encode_image_with_transformer(item["clip_cond"])
            ip_ = ip_embed.repeat_interleave(num_frames, dim=0)

            ip_img = model.get_first_stage_encoding(
                model.encode_first_stage(item["vae_cond"])
            )

            gd = rearrange(gd, "B F C H W -> (B F) C H W")

            latent_target_images = model.get_first_stage_encoding(
                model.encode_first_stage(gd)
            )

            if gd_type == "fusechannel":
                latent_target_images = rearrange(
                    latent_target_images, "(B F) C H W -> B F C H W", B=bs * 2
                )
                image_latent, xyz_latent = torch.chunk(latent_target_images, 2)
                fused_channel_latent = torch.cat((image_latent, xyz_latent), dim=-3)
                latent_target_images = rearrange(
                    fused_channel_latent, "B F C H W -> (B F) C H W"
                )

            if item.get("captions", None) is not None:
                caption_ls = np.array(item["caption"]).T.reshape((-1, BS)).squeeze()
                prompt_cond = model.get_learned_conditioning(caption_ls)
            elif item.get("caption", None) is not None:
                prompt_cond = model.get_learned_conditioning(item["caption"])
                prompt_cond = prompt_cond.repeat_interleave(num_frames, dim=0)
            else:
                prompt_cond = model.get_learned_conditioning(["3D assets."]).repeat(
                    BS, 1, 1
                )
            condition = {
                "context": prompt_cond,
                "ip": ip_,
                "ip_img": ip_img,
                "camera": camera_input,
            }

        with torch.autocast("cuda"), accelerator.accumulate(model):
            time_steps = torch.randint(0, model.num_timesteps, (BS,), device=device)
            noise = torch.randn_like(latent_target_images, device=device)
            # noise_img, _ = torch.chunk(noise, 2, dim=1)
            # noise = torch.cat((noise_img, noise_img), dim=1)
            x_noisy = model.q_sample(latent_target_images, time_steps, noise)
            output = unet(x_noisy, time_steps, **condition, num_frames=num_frames)
            reshaped_pred = output.reshape(bs, num_frames, *output.shape[1:]).permute(
                1, 0, 2, 3, 4
            )
            reshaped_noise = noise.reshape(bs, num_frames, *noise.shape[1:]).permute(
                1, 0, 2, 3, 4
            )
            true_pred = reshaped_pred[: num_frames - 1]
            fake_pred = reshaped_pred[num_frames - 1 :]
            true_noise = reshaped_noise[: num_frames - 1]
            fake_noise = reshaped_noise[num_frames - 1 :]
            loss = (
                torch.nn.functional.mse_loss(true_noise, true_pred)
                + torch.nn.functional.mse_loss(fake_noise, fake_pred) * 0
            )

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        total_step = global_step * config.total_batch_size
        if total_step > log_step:
            metrics = dict(
                loss=accelerator.gather(loss.detach().mean()).mean().item(),
                scale=(
                    accelerator.scaler.get_scale()
                    if accelerator.scaler is not None
                    else -1
                ),
            )
            log_step += config.log_interval
            if accelerator.is_main_process:
                logging.info(dct2str(dict(step=total_step, **metrics)))
                wandb.log(add_prefix(metrics, "train"), step=total_step)

        if total_step > save_step and accelerator.is_main_process:
            logging.info("saving done")
            torch.save(
                unet.state_dict(), osp.join(config.ckpt_root, f"unet-{total_step}")
            )
            save_step += config.save_interval
            logging.info("save done")

        if total_step > eval_step:
            logging.info("evaluationing")
            unet.eval()
            return_ls = evaluation()
            cur_eval_base = osp.join(config.eval_root, f"{total_step:07d}")
            os.makedirs(cur_eval_base, exist_ok=True)
            for item in return_ls:
                for i, im in enumerate(item["images"]):
                    im.save(
                        osp.join(
                            cur_eval_base,
                            f"{item['ident']}-{i:03d}-{accelerator.process_index}-.png",
                        )
                    )

            return_ls2 = evaluation2()
            cur_eval_base = osp.join(config.eval_root2, f"{total_step:07d}")
            os.makedirs(cur_eval_base, exist_ok=True)
            for item in return_ls2:
                for i, im in enumerate(item["images"]):
                    im.save(
                        osp.join(
                            cur_eval_base,
                            f"{item['ident']}-{i:03d}-{accelerator.process_index}-inthewild.png",
                        )
                    )
            eval_step += config.eval_interval
            logging.info("evaluation done")

        accelerator.wait_for_everyone()
        if total_step > config.max_step:
            break


if __name__ == "__main__":
    # load config from config path, then merge with cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/nf7_v3_SNR_rd_size_stroke.yaml"
    )
    parser.add_argument(
        "--logdir", type=str, default="train_logs", help="the dir to put logs"
    )
    parser.add_argument(
        "--resume_workdir", type=str, default=None, help="specify to do resume"
    )
    args, unk = parser.parse_known_args()
    print(args, unk)
    config = OmegaConf.load(args.config)
    if args.resume_workdir is not None:
        assert osp.exists(args.resume_workdir), f"{args.resume_workdir} not exists"
        config.config.workdir = args.resume_workdir
        config.config.resume = True
    OmegaConf.set_struct(config, True)  # prevent adding new keys
    cli_conf = OmegaConf.from_cli(unk)
    config = OmegaConf.merge(config, cli_conf)
    config = config.config
    OmegaConf.set_struct(config, False)
    config.logdir = args.logdir
    config.config_name = Path(args.config).stem

    train(config, unk)
