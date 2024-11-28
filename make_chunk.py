'''
Chunk make main function
'''
import os
import argparse
from subprocess import Popen

# Change to your cuda home path
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'

# Change to your /path/to/python
PYTHON = '/home/yudajiang/softwares/miniforge-pypy3/envs/crm/bin/python'

def main():
    '''
    Chunk make main function
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

    args = arg_parser.parse_args()

    p_ls = []
    try:
        with open('log.log', 'w', encoding='utf8') as log_f:
            for i in range(args.num_process):
                p = Popen([PYTHON, 'make_chunk_sub.py', '-e', args.exp_dir, '-c', args.config_name,
                        f'-p {args.num_process}',f"-i {i}"], stdout=log_f, stderr=log_f)
                p_ls.append(p)

            for p in p_ls:
                p.wait()
    except KeyboardInterrupt:
        # kill running processes
        for p in p_ls:
            p.kill()

if __name__ == '__main__':
    main()
