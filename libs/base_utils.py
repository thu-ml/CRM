import numpy as np
import cv2
import torch
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import os
import shutil
from absl import logging
import sys
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
import datetime
import os.path as osp
import torch.distributed as dist
import builtins
import accelerate
import wandb
import re
from diffusers.training_utils import EMAModel
from rich import print
        

def get_obj_from_str(string, reload=False):
    import importlib
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

        
def tensor_detail(t):
    assert type(t) == torch.Tensor
    print(f"shape: {t.shape} mean: {t.mean():.2f}, std: {t.std():.2f}, min: {t.min():.2f}, max: {t.max():.2f}")


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    if config.get("resume", False):
        print(f"resume from: {config.get('resume')}")
        if os.path.isfile(config.get("resume")):
            model.load_state_dict(torch.load(config["resume"], map_location="cpu"))
        elif os.path.isdir(config.get("resume")) and hasattr(model, "from_pretrained"):
            model.from_pretrained(config.get("resume"))
        else:
            raise Exception("could not resume")
    return model


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})



def copy_files_by_suffix(source_dir, target_dir, suffixes=[".py"], exclude_dirs=[]):
    # Walk through the directory tree
    for root, _, files in os.walk(source_dir):
        if any(exclude_dir in root for exclude_dir in exclude_dirs):
            continue
        for file in files:
            # Check if the file has one of the specified suffixes
            if any(file.endswith(suffix) for suffix in suffixes):
                # Construct the source and target paths
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)

                # Ensure the target directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Copy the file
                shutil.copyfile(source_path, target_path)



def find_latest_step(regex, ckpt_root):
    if not isinstance(regex, re.Pattern):
        regex = re.compile(regex)
    ints = []
    for file in os.listdir(ckpt_root):
        if re.match(regex, file):
            ints.append(int(re.findall(r'\d+', file)[0]))
    if len(ints) == 0:
        raise FileNotFoundError(f"no file match {regex} in {ckpt_root}")
    return max(ints)
    
    
def resume_from_workdir(config, accelerator, model_context, ema_context):
    if config.get("resume", False):
        with PrintContext(f"resume from {config.workdir}", accelerator.is_main_process):
            for name in config.save_models:
                max_step = find_latest_step(f"{name}-(\d+).pt", config.ckpt_root)
                print(f"resume from {name}-{max_step}.pt")
                model_context[name].load_state_dict(
                    torch.load(osp.join(config.ckpt_root, f"{name}-{max_step}.pt"),  
                               map_location="cpu")
                )
            for k, ema in ema_context.items():
                max_step = find_latest_step(f"{k}-ema-(\d+).pt", config.ckpt_root)
                print(f"resume from {k}-ema-{max_step}.pt")
                ema.load_state_dict(
                    torch.load(osp.join(config.ckpt_root, f"{k}-ema-{max_step}.pt"),
                                 map_location="cpu")
                )
                ema.to(accelerator.device)
        return max_step
    else:
        return 0


def get_model_context(models, device, dtype):
    model_context = dict()
    for key, model_config in models.items():
        model = instantiate_from_config(model_config)
        if hasattr(model, "device"):
            try:
                model.device = device
            except Exception as e:
                print(e)
                print('passing set device')
        if "t5" in type(model).__name__.lower() and isinstance(model, nn.Module):
            # T5 model has a bug that it when using fp16
            print(f"{'passing t5 model':-^72}")
            model_context[key] = model.to(device)
            continue
        if isinstance(model, nn.Module):
            model_context[key] = model.to(device=device, dtype=dtype)
        else:
            model_context[key] = model
    model_context["device"] = device
    model_context["dtype"] = dtype
    return model_context

def get_ema_context(model_context, emas):
    """given config of ema models and model context, return an ema_context
        contains all ema model in the current train process

    Args:
        model_context (dict): dict of names, point to pytroch models
        emas (dict): dict of name, point to ema model, name was same with
    """
    ema_context = dict()
    if emas is None:
        return ema_context
    for ema_item in emas:
        name = ema_item["name"]
        ema_context[name] = EMAModel(model_context[name].parameters(), **ema_item.params)
    return ema_context


def get_data_context(data, accelerator=None):
    data_context = dict()
    for key, data_config in data.items():
        dataset = instantiate_from_config(data_config.dataset)
        if data_config.get("distributed_sampler", False):
            sampler_cls = get_obj_from_str(data_config.distributed_sampler.target)
            distributed_sampler = sampler_cls(
                dataset,
                num_replicas=accelerator.num_processes if accelerator is not None else 1,
                rank=accelerator.process_index if accelerator is not None else 0,
                **data_config.distributed_sampler.params
            )
            dataloader = DataLoader(dataset, sampler=distributed_sampler, **data_config.dataloader)
        else:
            dataloader = DataLoader(dataset, **data_config.dataloader)
        data_context[key] = dataloader
        data_context[key + "_generator"] = get_data_generator(dataloader, accelerator.is_main_process if accelerator is not None else True, key)
        data_context[key + "_dataset"] = dataset
    return data_context


class Unimodel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._module_list = nn.ModuleList(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)


def config_optimizer(model_context, optimizer_models, default_opt_params):
    """
    model_context: dict of model instances
    optimizer_models: list of dict, each dict contains model name and modules
    default_opt_params: dict of default optimizer parameters
    """
    default_opt_params = dict(default_opt_params)
    param_groups = []
    for model_config in optimizer_models:
        model = model_context[model_config["name"]]
        if model_config.get("modules", None) is None:  # all model when no sub modules specified
            model.requires_grad_(True)
            print(f"using all modules of {model_config['name']}")
            para_dict = default_opt_params.copy()
            opt_params = model_config.get("opt_params", dict())
            para_dict.update(opt_params)
            para_dict["params"] = list(model.parameters())
            param_groups.append(para_dict)
        else:
            model.requires_grad_(False)
            for module_config in model_config["modules"]:
                para_dict = default_opt_params.copy()
                params = []
                for name, param in model.named_parameters():
                    if module_config["name"] in name:
                        print(name)
                        param.requires_grad = True
                        params.append(param)
                para_dict["params"] = params
                opt_params = model_config.get("opt_params", dict())
                para_dict.update(opt_params)
                param_groups.append(para_dict)
    return param_groups


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def get_hparams(input_args=None):
    argv = sys.argv if input_args is None else input_args
    lst = []
    for i in range(len(argv)):
        if argv[i].startswith('config.'):
            hparam_full, val = argv[i].split('=')
            hparam = hparam_full.split('.')[-1]
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def add_prefix(dct, prefix):
    return {f'{prefix}/{key}': val for key, val in dct.items()}


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def param_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

class PrintContext(object):
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose: print(f'{self.name} processing...')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose: print(f'{self.name} done')

def time_to_tensor(now: datetime.datetime):
    return torch.tensor([now.year, now.month, now.day, now.hour, now.minute, now.second], dtype=torch.long)

def tensor_to_time(t: torch.Tensor):
    return datetime.datetime(*t.tolist())



def setup(config, unk):
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    
    # sync time for all processes
    g_handler = dist.new_group(backend='gloo')
    now = time_to_tensor(datetime.datetime.now())
    dist.broadcast(now, src=0, group=g_handler)
    now = tensor_to_time(now).strftime("%Y-%m-%dT%H-%M-%S")
    print("unknow args: ", unk, get_hparams(unk))

    if config.get("workdir", None) is None:
        config.workdir = osp.join(config.logdir, f"{config.config_name}-{get_hparams(unk)}-{now}")
    print(f"{'workdir: ' + config.workdir:-^72}")
    config.ckpt_root = osp.join(config.workdir, 'ckpts')
    config.eval_root = osp.join(config.workdir, "eval")
    config.eval_root2 = osp.join(config.workdir, "eval2")
    
    if accelerator.is_main_process:
        os.makedirs(config.workdir, exist_ok=True)
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.eval_root, exist_ok=True)
        os.makedirs(config.eval_root2, exist_ok=True)
    
        config.meta_dir = osp.join(config.workdir, f"meta-{now}")
        copy_files_by_suffix(os.getcwd(), config.meta_dir, exclude_dirs=[config.logdir], suffixes=[".py", ".yaml"])

        with open(osp.join(config.meta_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config))

    wandb.init(dir=os.path.abspath(config.workdir), project=config.project, config=dict(config),
               name=config.wandb_run_name, job_type='train', mode=config.wandb_mode, group="DDP")
    if accelerator.is_main_process:
        set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        print(OmegaConf.to_yaml(config))
    else:
        set_logger(log_level='error')
        builtins.print = lambda *args: None

    assert not ('total_batch_size' in config and 'batch_size' in config)
    if 'total_batch_size' not in config:
        config.total_batch_size = config.batch_size * accelerator.num_processes
    if 'batch_size' not in config:
        assert config.total_batch_size % accelerator.num_processes == 0
        config.batch_size = config.total_batch_size // accelerator.num_processes
    if 'total_logical_batch_size' not in config:
        config.total_logical_batch_size = config.total_batch_size * config.gradient_accumulation_steps

    logging.info(f'Run on {accelerator.num_processes} devices')

    return accelerator, device


def get_data_generator(loader, enable_tqdm, desc):
    while True:
        for data in tqdm(loader, disable=not enable_tqdm, desc=desc):
            yield data


def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = ((original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image

def add_stroke(img, color=(255, 255, 255), stroke_radius=3):
    # color in R, G, B format
    if isinstance(img, Image.Image):
        assert img.mode == "RGBA"
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    else:
        assert img.shape[2] == 4
    gray = img[:,:, 3]
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    res = cv2.drawContours(img, contours,-1, tuple(color)[::-1] + (255,), stroke_radius)
    return Image.fromarray(cv2.cvtColor(res,cv2.COLOR_BGRA2RGBA))

def make_blob(image_size=(512, 512), sigma=0.2):
    """
    make 2D blob image with:
    I(x, y)=1-\exp \left(-\frac{(x-H / 2)^2+(y-W / 2)^2}{2 \sigma^2 HS}\right)
    """
    import numpy as np
    H, W = image_size
    x = np.arange(0, W, 1, float)
    y = np.arange(0, H, 1, float)
    x, y = np.meshgrid(x, y)
    x0 = W // 2
    y0 = H // 2
    img = 1 - np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2 * H * W))
    return (img * 255).astype(np.uint8)