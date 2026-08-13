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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "逐塊訓練 CityGS 分區：對每個 block 各跑一次 main.py fit，序列執行、等 GPU 空出來才啟動下一塊。\n"
            "\n"
            "範例（我方 depth-init 流程）：\n"
            "  python utils/train_citygs_partitions.py -n sh3_4x4_aerial \\\n"
            "      --init_mode depth --depth_init_dir data/matrix_city/aerial/train/block_all/depth_init_4x4\n"
            "\n"
            "  -n 只給【檔名】不含目錄與 .yaml；它同時是輸出目錄名（outputs/<n>/）。\n"
            "  塊數由 config 的 data.parser.init_args.block_dim 決定，不由參數指定。\n"
            "  已有 checkpoint 的塊會自動跳過，所以中斷後重跑即可續作。\n"
            "\n"
            "未被本程式辨識的參數會原封不動傳給 main.py fit，例如：\n"
            "  ... -n sh3_4x4_aerial --model.density.init_args.cap_max 2000000\n"
        ),
        epilog=(
            "═══ 常用的透傳參數（本程式不解析，原樣交給 main.py fit）═══\n"
            "\n"
            "  --model.density.init_args.cap_max INT\n"
            "      顆數硬上限，主要的 VRAM 槓桿。實際終值約 0.9*cap（撞頂後最後一次剪枝砍 10%）。\n"
            "      每顆位元組 = 4*F*4(參數+梯度+2 Adam) + 約 978(渲染)；F=25(SB)/59(SH3)。\n"
            "      6GB 可用約 5.6G ⇒ SB 上限約 4.06M、SH3 約 3.13M。實測 SB 3.78M 活、4.00M OOM。\n"
            "      ⚠ 不要用 tools/calibrate_block_caps.py 定它：那組常數的形狀就錯（R^2=0.066）。\n"
            "\n"
            "  --model.metric.init_args.opacity_reg FLOAT   [0 ~ 0.01，我方最佳 0.002]\n"
            "      不透明度 L1。0.002 在 900k 顆上比 0 高 0.74 dB；0.007 反而掉 0.53 dB。\n"
            "      ⚠ 它會抬高 VRAM（壓低 opacity ⇒ 每像素混合更多層）：同 900k 顆 0→0.002 多吃 1.81 GB。\n"
            "\n"
            "  --model.density.init_args.screen_size_prune_px INT   [-1=關，我方用 300]\n"
            "      螢幕足跡超過此像素數的粒子視為怪物並剪除。\n"
            "\n"
            "  --model.gaussian.init_args.optimization.means_lr_scheduler.lr_final FLOAT\n"
            "      位置學習率的終值（出廠 6.4e-7 = 初值的 1/100）。提到 6.4e-6 使積分變 1.8 倍，\n"
            "      實測 +0.16 dB 且不花額外時間；但它會磨掉 blur split 帶來的細節（兩者互相抵消）。\n"
            "\n"
            "  --model.density.init_args.blur_split_budget FLOAT   [0=關，建議 0.3]\n"
            "      每次 densify 有多少比例的預算投給「獨自糊住一大片」的粒子（Mini-Splatting 式\n"
            "      under-reconstruction 判準）。用【佔比】而非倍率，因為候選只佔族群 0.01%，\n"
            "      任何倍率都會被淹沒。在 SB 上值 LPIPS -0.013、紋理比 +0.019；在 SH3 上失效。\n"
            "\n"
            "  --model.density.init_args.blur_split_threshold FLOAT   [預設 288 = 2e-4*900*1600]\n"
            "      上述判準的門檻，單位是像素。只有與分數的比值被使用，等於定義「超標一個單位」。\n"
            "\n"
            "  --model.density.init_args.long_axis_spread FLOAT   ⛔ 保持 0\n"
            "      沿長軸散開子代。已實測證偽（-0.13 dB = 2.4 倍噪音）：它破壞 MCMC Eq.9 的共位前提。\n"
            "      程式保留僅作為相關工作的反例。\n"
            "\n"
            "  --model.renderer.init_args.contribution_prune_until_iter INT   [-1=跟隨 densify_until_iter]\n"
            "      收割（貢獻度剪枝）停止的步數。設 -1 以外的值才能表達「densify 停了但繼續剪」。\n"
            "\n"
            "═══ 注意事項 ═══\n"
            "\n"
            "  ⚠ 不要用 --config：與 --config_name / --config_dir 前綴相同會被判定歧義。\n"
            "    本程式已關閉前綴縮寫（allow_abbrev=False），打錯的參數會直接報錯而非被靜默吞掉。\n"
            "  ⚠ 動輒數天的跑次請先加 --dry-run：參數錯誤會在第一秒就死掉，而你可能幾小時後才發現。\n"
            "  ⚠ 排程請用 scripts/runner.sh + scripts/queue.txt，不要自己另外寫 watcher。\n"
        ),
        allow_abbrev=False,   # 關掉前綴縮寫：避免 --config 這類歧義，也避免打錯字被靜默接受
    )
    parser.add_argument(
        "--config_name", "-n", type=str, required=True, metavar="NAME",
        help=("config 檔名，不含目錄與副檔名（實際讀 <--config_dir>/NAME.yaml）。"
              "同時作為輸出目錄名 outputs/NAME/。例：sh3_4x4_aerial"),
    )
    parser.add_argument(
        "--config_dir", "-c", type=str, default="./configs", metavar="DIR",
        help="config 所在目錄（預設 %(default)s）",
    )
    parser.add_argument(
        "--project_name", "-p", type=str, default="", metavar="PROJ",
        help="wandb 專案名；留空則用 TensorBoard（預設留空）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help=("只列出要訓練哪些塊與各塊的 initialize_from，不實際啟動。"
              "投入多日跑次前務必先跑一次——參數錯誤會在第一秒就死掉，而你可能幾小時後才發現"),
    )
    parser.add_argument(
        "--init_mode", choices=["depth", "coarse", "config"], default=None,
        help=(
            "每塊高斯的初始化來源。"
            "depth：從 --depth_init_dir 取 block_<id>.ply（給了 --depth_init_dir 時的預設，我方主線用這個）；"
            "coarse：所有塊共用 --coarse_ckpt 的同一個 checkpoint（上游 CityGS 流程）；"
            "config：照 YAML 裡的 initialize_from 原樣使用（其餘情況的預設）"
        ),
    )
    parser.add_argument(
        "--depth_init_dir", type=str, default=None, metavar="DIR",
        help=("逐塊 depth-init PLY 所在目錄，需含 block_0.ply ... block_<N-1>.ply（配合 --init_mode depth）。"
              "⚠ 目錄要與 config 的 block_dim 相符：5x5 用 depth_init/、4x4 用 depth_init_4x4/、3x3 用 depth_init_3x3/。"
              "用錯目錄不會報錯，只會拿到覆蓋範圍不對的初始化"),
    )
    parser.add_argument(
        "--coarse_ckpt", type=str, default=None, metavar="CKPT",
        help=("全域 coarse 模型的 .ckpt 路徑，作為所有塊的 initialize_from（配合 --init_mode coarse）。"
              "要指到 .ckpt 檔本身，不是目錄"),
    )
    parser.add_argument(
        "--blocks", type=int, nargs="+", default=None, metavar="ID",
        help=("只訓練指定的 block ID，0 起算（例：--blocks 0 3 7）。"
              "預設訓練全部；已有 checkpoint 的塊一律跳過"),
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