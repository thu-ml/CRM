'''
Chunk make subprocess (script for debug use)
'''
import os
import json
import argparse

from datasets.fast_dataset import FastDataset

# Change to your cuda home path
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'


def main():
    '''
    Chunk Make subprocess (for debug use)
    '''
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--exp_dir', '-e',
        default='configs/',
        help='This directory should include experiment specifications in "specs.json" ',
    )
    arg_parser.add_argument(
        '--config_name', '-c',
        default='specs_clear.json',
        help='config filename',
    )
    arg_parser.add_argument(
        '-p', '--num_process',
        default=1,
        type=int,
        help='Number of processes to use for data loading.',
    )
    arg_parser.add_argument(
        '-i', '--process_index',
        default=0,
        type=int,
        help='Index of the current process.',
    )

    args = arg_parser.parse_args()
    config_path = os.path.join(args.exp_dir, args.config_name)

    with open(config_path, encoding='utf8') as f:
        specs = json.load(f)

    input_specs = specs['Input']
    train_specs = specs['Train']
    dataset = FastDataset(input_specs, train_specs, None, None, 0, mode='all')

    dataset.num_process = args.num_process
    dataset.process_index = args.process_index
    dataset.write_chunks(dataset.parquet_dir, dataset.chunk_size)


if __name__ == '__main__':
    main()
