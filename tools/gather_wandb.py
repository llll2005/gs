import os
import sys
import yaml
import torch
import numpy as np
import subprocess
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor


def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_path', type=str, help='path of output folder', default=None)
    args = parser.parse_args(sys.argv[1:])
    
    target_path = f"{args.output_path}_wandb_logs"
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    subfolders = [f.path for f in os.scandir(args.output_path) if f.is_dir()]
    for subfolder in tqdm(subfolders, desc="Gathering wandb logs"):
        subfolder_name = os.path.basename(subfolder)
        target_subfolder = os.path.join(target_path, subfolder_name)

        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder, exist_ok=True)

        wandb_path = os.path.join(subfolder, 'wandb')
        if os.path.exists(wandb_path):
            subprocess.run(['cp', '-r', wandb_path, target_subfolder])
        else:
            print(f"No wandb logs found in {subfolder}, skipping.")

if __name__ == "__main__":
    main()