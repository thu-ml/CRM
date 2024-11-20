import os
import json
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
import argparse
# import nvdiffrast.torch as dr
from itertools import cycle
from tqdm import tqdm
import torch.distributed as dist

from datasets.memory_dataset import MemoryDataset
from datasets.filesystem_dataset import FilesystemDataset
from datasets.fast_dataset import FastDataset
# from util.common import pcd_preprocess
from accelerate import Accelerator
from util import utils
from subprocess import Popen, PIPE


def train(specs, config_path):
    input_specs = specs['Input']
    task_mode = input_specs['task_mode']
    tet_grid_size = input_specs['tet_grid_size']
    dataset_type = input_specs.get('dataset_type', None)

    train_specs = specs['Train']

    device = 'cuda:0'
    # glctx = dr.RasterizeCudaContext()
    glctx = None

    dataset, dataset_val_all = [None] * 2

    if dataset_type == 'memory' or dataset_type is None:
        dataset = MemoryDataset(input_specs, train_specs, glctx, tet_grid_size, device, task_mode=task_mode)

    elif dataset_type == 'filesystem':
        dataset = FastDataset(input_specs, train_specs, glctx, device, 0, mode='all')
    return dataset


def main():
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
        "-p", "--num_process",
        default=1,
        type=int,
        help="Number of processes to use for data loading.",
    )
    
    arg_parser.add_argument(
        "-i", "--process_index",
        default=0,
        type=int,
        help="Index of the current process.",
    )

    args = arg_parser.parse_args()
    config_path = os.path.join(args.exp_dir, args.config_name)

    specs = json.load(open(config_path))
    dataset = train(specs, config_path)
    
    dataset.num_process = args.num_process
    dataset.process_index = args.process_index
    dataset._write_chunks(dataset.parquet_dir, dataset.chunk_size)


if __name__ == '__main__':
    main()