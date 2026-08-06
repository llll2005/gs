import os
import argparse
import yaml
import concurrent
import add_pypath
import subprocess
import traceback
import time
import selectors
import py3nvml
import numpy as np
import functools
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from auto_hyper_parameter import auto_hyper_parameter, to_command_args
from argparser_utils import split_stoppable_args, parser_stoppable_args
from internal.utils.general_utils import parse

def get_project_output_dir_by_name(project_name: str) -> str:
    return os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), "outputs", project_name)

def srun_output_dir(project_name: str) -> str:
    return os.path.join(get_project_output_dir_by_name(project_name), "srun-outputs")

def run_subprocess(args, output_redirect) -> int:
    sel = selectors.DefaultSelector()

    with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        sel.register(p.stdout, selectors.EVENT_READ)
        sel.register(p.stderr, selectors.EVENT_READ)

        while True:
            if len(sel.get_map()) == 0:
                break

            events = sel.select()
            for key, mask in events:
                line = key.fileobj.readline()
                if len(line) == 0:
                    sel.unregister(key.fileobj)
                    continue
                output_redirect(line.decode("utf-8").rstrip("\n"))
        p.wait()
        return p.returncode

def train_a_partition(
        config_args,
        extra_training_args,
        srun_args,
        partition_idx,
        gpu_id,
    ):
    config_file = os.path.join(config_args.config_dir, f"{config_args.config_name}.yaml")
    project_name = config_args.project_name
    dry_run = config_args.dry_run

    # resolve init_mode (auto-detect when not explicit)
    init_mode = config_args.init_mode
    if init_mode is None:
        if config_args.depth_init_dir is not None:
            init_mode = "depth"
        elif config_args.coarse_ckpt is not None:
            init_mode = "coarse"
        else:
            init_mode = "config"

    # build args
    # basic
    args = [
        "python",
        "main.py", "fit",
        "--config", config_file,
        "--data.parser.block_id", str(partition_idx),
    ]

    # initialize_from based on init_mode
    if init_mode == "depth":
        ply_path = os.path.join(config_args.depth_init_dir, f"block_{partition_idx}.ply")
        if os.path.exists(ply_path):
            args += ["--model.initialize_from", ply_path]
        else:
            print(f"[warn] depth PLY not found for block {partition_idx}: {ply_path}")
    elif init_mode == "coarse":
        if config_args.coarse_ckpt is not None:
            args += ["--model.initialize_from", config_args.coarse_ckpt]
        else:
            print("[warn] --init_mode coarse requires --coarse_ckpt")
    # "config": no override, use initialize_from from YAML as-is

    # extra
    args += extra_training_args

    experiment_name = config_args.config_name
    args += ["-n={}".format(experiment_name)]
    if project_name:
        args += ["--project", project_name, "--logger", "wandb"]
    else:
        args += ["--logger", "tensorboard"]

    def ts():
        return time.strftime('%Y-%m-%d %H:%M:%S')

    print_func = lambda msg: print(f"[{ts()}] #{partition_idx}: {msg}")
    run_func = functools.partial(subprocess.run, env=dict(**os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id)))
    if len(srun_args) > 0:
        def run_with_tqdm_write(args):
            return run_subprocess(args, lambda i: tqdm.write(f"[{ts()}] #{partition_idx}: {i}"))

        run_func = run_with_tqdm_write

        output_filename = os.path.join(srun_output_dir(config_args.config_name), "block_{}.txt".format(partition_idx))
        args = [
            "srun",
            "--output={}".format(output_filename),
            "--job-name={}-{}".format(config_args.project_name, experiment_name),
        ] + srun_args + args

    ret_code = -1
    if dry_run:
        print("  " + " \\\n  ".join(args))
    else:
        try:
            print_func(f"starting (GPU {gpu_id})")
            ret_code = run_func(args)
            print_func(f"done (exit code {ret_code.returncode if hasattr(ret_code, 'returncode') else ret_code})")
        except KeyboardInterrupt as e:
            raise e
        except:
            traceback.print_exc()

    return partition_idx, ret_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", "-n", type=str, required=True)
    parser.add_argument("--config_dir", "-c", type=str, default="./configs")
    parser.add_argument("--project_name", "-p", type=str, default="", help="wandb project name; omit to use TensorBoard instead")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--init_mode", choices=["depth", "coarse", "config"], default=None,
        help=(
            "depth : per-block PLY from --depth_init_dir (default when --depth_init_dir given); "
            "coarse: same checkpoint for all blocks from --coarse_ckpt; "
            "config: use initialize_from from YAML as-is (default otherwise)"
        ),
    )
    parser.add_argument("--depth_init_dir", type=str, default=None,
                        help="directory of per-block depth-init PLYs (used with --init_mode depth)")
    parser.add_argument("--coarse_ckpt", type=str, default=None,
                        help="coarse checkpoint path used as initialize_from for all blocks (--init_mode coarse)")
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=None,
        help="specific block IDs to train (e.g. --blocks 0 3 7); default: all blocks",
    )

    args, training_and_srun_args = parser_stoppable_args(parser)
    training_args, srun_args = split_stoppable_args(training_and_srun_args)

    config_path = os.path.join(args.config_dir, f"{args.config_name}.yaml")
    with open(config_path, 'r') as f:
        config = parse(yaml.load(f, Loader=yaml.FullLoader))
    num_blocks = config.data.parser.init_args.block_dim[0] * config.data.parser.init_args.block_dim[1]

    block_id_list = args.blocks if args.blocks is not None else list(range(num_blocks))
    print(f"Blocks to train: {block_id_list} / {num_blocks} total")

    def is_block_complete(config_name, block_id):
        ckpt_dir = os.path.join(get_project_output_dir_by_name(config_name), "blocks", f"block_{block_id}", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return False
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and "xyz_rgb" not in f]
        return len(ckpts) > 0

    if len(srun_args) == 0:
        with ProcessPoolExecutor(max_workers=len(block_id_list)) as executor:
            for block_id in block_id_list:
                if is_block_complete(args.config_name, block_id):
                    print(f"[skip] block_{block_id} already has checkpoints, skipping.")
                    continue

                # Single-GPU note: get_free_gpus(max_procs=0) treats a GPU as free only
                # when ZERO compute procs run on it, so blocks strictly serialize (the
                # previous block must fully exit before the next is submitted — this is
                # what keeps the 6GB card from OOMing on two co-located blocks). The GPU
                # WILL free up once the running block finishes, so we wait indefinitely.
                # (The old `fail_cnt > 90` hard exit() capped the wait at 3h and silently
                # killed the whole run — including all not-yet-submitted blocks — whenever
                # a single block trained longer than 3h. That is why block_16 vanished when
                # block_7's settings pushed it past 3h.)
                gpu_available = False
                fail_cnt = 0
                wait_start = time.time()
                while not gpu_available:
                    free_gpus = py3nvml.get_free_gpus()
                    if sum(free_gpus) > 0:
                        gpu_available = True
                    else:
                        if fail_cnt == 0:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] GPU busy, waiting for block to finish...", flush=True)
                        elif fail_cnt % 5 == 0:
                            elapsed = int((time.time() - wait_start) / 60)
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] still waiting... ({elapsed}m elapsed)", flush=True)
                        fail_cnt += 1
                        time.sleep(120)

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [submit] block_{block_id} → GPU {np.argmax(free_gpus)}")
                executor.submit(train_a_partition, args, training_args, srun_args, block_id, np.argmax(free_gpus))

                time.sleep(90)  # let the subprocess start and claim GPU before next check
    else:
        print("SLURM mode enabled")
        trainable_partition_idx_list = block_id_list
        total_trainable_partitions = len(trainable_partition_idx_list)

        with ThreadPoolExecutor(max_workers=total_trainable_partitions) as tpe:
            futures = [tpe.submit(
                train_a_partition,
                args,
                training_args,
                srun_args,
                i,
            ) for i in trainable_partition_idx_list]
            finished_count = 0
            with tqdm(
                    concurrent.futures.as_completed(futures),
                    total=total_trainable_partitions,
                    miniters=1,
                    mininterval=0,  # keep progress bar updating
                    maxinterval=0,
            ) as t:
                for future in t:
                    finished_count += 1
                    try:
                        finished_idx, ret_code = future.result()
                    except KeyboardInterrupt as e:
                        raise e
                    except:
                        traceback.print_exc()
                        continue
                    tqdm.write("[{}] #{} exited with code {} | {}/{}".format(
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                        finished_idx,
                        ret_code,
                        finished_count,
                        total_trainable_partitions,
                    ))