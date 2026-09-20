"""峰值記錄（2026-09-17）：**只在設了 CITYGS_PEAK_OUT 時作用**，不 import torch、不改任何行為。

行程結束時讀 torch.cuda.max_memory_allocated / max_memory_reserved，寫一行 TSV：
  pid  峰值配置MiB  峰值保留MiB  存活秒數  argv
用途：官方 cityGS_origin 沒有我方台帳，又不能改官方原始碼 => 經 PYTHONPATH 外掛。
DataLoader worker 沒初始化 CUDA，會自動略過；行程被殺（OOM 等）就不會有這一行。
"""
import atexit
import os
import sys
import time


def _chain_original():
    # 環境原本若有 sitecustomize，照樣執行它（避免被這一份遮蔽）
    here = os.path.dirname(os.path.abspath(__file__))
    for p in sys.path:
        try:
            if not p or os.path.abspath(p) == here:
                continue
            f = os.path.join(p, "sitecustomize.py")
            if os.path.isfile(f):
                import importlib.util as u
                s = u.spec_from_file_location("_orig_sitecustomize", f)
                m = u.module_from_spec(s)
                s.loader.exec_module(m)
                return
        except Exception:
            return


_chain_original()

_out = os.environ.get("CITYGS_PEAK_OUT")
if _out:
    _t0 = time.time()

    def _dump():
        t = sys.modules.get("torch")
        if t is None:
            return
        try:
            if not t.cuda.is_available() or not t.cuda.is_initialized():
                return
            a = t.cuda.max_memory_allocated() / 2 ** 20
            r = t.cuda.max_memory_reserved() / 2 ** 20
            argv = " ".join(sys.argv)[:300]
            with open(_out, "a") as f:
                f.write(f"{os.getpid()}\t{a:.0f}\t{r:.0f}\t{time.time() - _t0:.0f}\t{argv}\n")
        except Exception:
            pass

    atexit.register(_dump)
