import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.1"
import argparse
from subprocess import Popen, PIPE


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        default="configs/",
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

    args = arg_parser.parse_args()

    python = "/home/yudajiang/softwares/miniforge-pypy3/envs/crm/bin/python"
    p_ls = []
    try:
        log_f = open("log.log", "w")
        for i in range(args.num_process):
            p = Popen([python, 'make_chunk_sub.py', '-e', args.exp_dir, '-c', args.config_name, f'-p {args.num_process}',f"-i {i}"], stdout=log_f, stderr=log_f)
            p_ls.append(p)
            
        [p.wait() for p in p_ls]
    except:
        for p in p_ls:
            p.kill()

if __name__ == '__main__':
    main()