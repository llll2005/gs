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


# ── 訓練過程紀錄（2026-09-24）：**只在設了 CITYGS_TRAINLOG_OUT 時作用** ─────────────────
# 用途：官方參考線（未修改的 cityGS_origin）沒有我方台帳；使用者允許「我方的 log/計數器」
#   外掛，但官方原始碼一行都不能改 ⇒ 在 import 時替 lightning 的 Trainer 多掛一個唯讀 callback。
# 每 CITYGS_TRAINLOG_EVERY（預設 100）個 batch 寫一行 TSV：
#   batch  global_step  牆鐘秒  區間it/s  N  目前配置MiB  目前保留MiB  累計峰值配置MiB  累計峰值保留MiB
# 結束寫 END／例外寫 DIED＋例外字串。
# ⚠ 唯讀：不 reset 峰值統計（那會改掉上面 atexit 的整段峰值）、不同步、不碰 RNG、不改任何張量。
#   N 只讀 get_xyz.shape[0]（參數的形狀），不做計算。
_tl = os.environ.get("CITYGS_TRAINLOG_OUT")
if _tl:
    import importlib.abc
    import importlib.machinery

    _every = int(os.environ.get("CITYGS_TRAINLOG_EVERY", "100"))

    def _w(line):
        try:
            new = not os.path.exists(_tl)
            with open(_tl, "a") as f:
                if new:
                    f.write("batch\tglobal_step\twall_s\tit_s\tN\talloc_MiB\treserved_MiB\tpeak_alloc_MiB\tpeak_reserved_MiB\n")
                f.write(line + "\n")
        except Exception:
            pass

    def _make_cb(Callback):
        class _TrainLog(Callback):
            def __init__(self):
                self.k = 0
                self.t0 = self.tl = time.time()
                self.kl = 0

            def _row(self, trainer, pl_module, tag=""):
                t = sys.modules["torch"]
                now = time.time()
                its = (self.k - self.kl) / max(now - self.tl, 1e-9)
                self.kl, self.tl = self.k, now
                try:
                    n = int(pl_module.gaussian_model.get_xyz.shape[0])
                except Exception:
                    n = -1
                m = [0.0] * 4
                try:
                    c = t.cuda
                    m = [c.memory_allocated() / 2 ** 20, c.memory_reserved() / 2 ** 20,
                         c.max_memory_allocated() / 2 ** 20, c.max_memory_reserved() / 2 ** 20]
                except Exception:
                    pass
                _w(f"{self.k}\t{trainer.global_step}\t{now - self.t0:.1f}\t{its:.3f}\t{n}\t"
                   + "\t".join(f"{v:.0f}" for v in m) + (f"\t{tag}" if tag else ""))

            def on_train_start(self, trainer, pl_module):
                self.t0 = self.tl = time.time()
                self._row(trainer, pl_module, "START")

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.k += 1
                if self.k % _every == 0:
                    self._row(trainer, pl_module)

            def on_train_end(self, trainer, pl_module):
                self._row(trainer, pl_module, "END")

            def on_exception(self, trainer, pl_module, exception):
                self._row(trainer, pl_module, "DIED " + repr(exception)[:300].replace("\t", " ").replace("\n", " "))

        return _TrainLog

    def _patch(mod):
        # ⚠ 不包 Trainer.__init__：LightningCLI/jsonargparse 會讀它的簽名來產生 CLI 參數。
        #   掛在 callback connector 上：它把使用者的 callbacks 整理成 trainer.callbacks 之後，再多加一個。
        from lightning.pytorch.callbacks import Callback
        C = mod._CallbackConnector
        orig = C.on_trainer_init
        CB = _make_cb(Callback)

        def on_trainer_init(self, *a, **kw):
            r = orig(self, *a, **kw)
            self.trainer.callbacks.append(CB())
            print(f"[trainlog] ✅ 已掛上唯讀紀錄 callback -> {_tl}（每 {_every} batch）", file=sys.stderr)
            return r

        C.on_trainer_init = on_trainer_init

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "lightning.pytorch.trainer.connectors.callback_connector":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            _orig_exec = spec.loader.exec_module

            def exec_module(module):
                _orig_exec(module)
                try:
                    _patch(module)
                except Exception as e:
                    print(f"[trainlog] ⚠⚠ 掛不上 callback：{e!r}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
