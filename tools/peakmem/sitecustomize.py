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


# ── 訓練過程紀錄（2026-09-24）：官方參考線用，**只在設了 CITYGS_TRAINLOG_OUT 時作用** ─────────
# 使用者允許「我方的 log／計數器」外掛，但官方原始碼一行都不能改 ⇒ 全部在 import 時掛上：
#   ① 訓練曲線  CITYGS_TRAINLOG_OUT：每 CITYGS_TRAINLOG_EVERY（預設 100）batch 一行
#        batch global_step wall_s it_s N alloc_MiB reserved_MiB peak_alloc_MiB peak_reserved_MiB [START/END/DIED]
#   ② churn     CITYGS_CHURN_OUT：每次加顆／剪枝一行（官方 density controller 與 trim 都走同兩個函式）
#        global_step phase added removed N_after
#        phase = clone／split（split 會先加 2k 再剪掉 k 個母體，removed 記的就是那 k）／
#                cull（opacity／螢幕尺寸／世界尺寸剪枝）／trim（renderer 的貢獻度剪枝，含 step 1 起始 trim）
#        另有 reset_opacity 事件（added=removed=0）
#   ④ 現況檔    train_status.txt（寫在跑次輸出目錄 trainer.default_root_dir；另設 CITYGS_STATUS_OUT 時再寫一份）
#        每 100 batch 覆寫：步數/總步數、N、it/s、預估剩餘、VRAM、churn 累計與最近一次事件、最近的 val PSNR
#   ③ 我方台帳  CITYGS_LEDGER：logs/quad_progress.log 的 START/DONE/DIED，格式與 internal/callbacks.py 相同
# ⚠ 唯讀：不 reset 峰值統計、不碰 RNG、不改任何張量或回傳值。churn 的 mask.sum().item() 只在
#   剪枝事件發生時同步一次（本來就要同步做索引），不改數值。
_tl = os.environ.get("CITYGS_TRAINLOG_OUT")
if _tl:
    import importlib.abc
    import importlib.machinery

    _every = int(os.environ.get("CITYGS_TRAINLOG_EVERY", "100"))
    _churn = os.environ.get("CITYGS_CHURN_OUT")
    _ledger = os.environ.get("CITYGS_LEDGER")
    _S = {"step": 0, "phase": [], "tot": {}, "last_ev": ""}
    _status_extra = os.environ.get("CITYGS_STATUS_OUT")

    def _append(path, header, line):
        if not path:
            return
        try:
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new and header:
                    f.write(header + "\n")
                f.write(line + "\n")
        except Exception:
            pass

    def _run_name():
        a = sys.argv
        name = blk = None
        for j, x in enumerate(a[:-1]):
            if x in ("-n", "--name"):
                name = a[j + 1]
            if x == "--data.parser.block_id":
                blk = a[j + 1]
        r = "OFF:" + (name or "run").replace("citygsv2_mc_aerial_", "")
        return r + (f"/b{blk}" if blk is not None else "")

    def _progress(kind, detail):
        if not _ledger:
            return
        try:
            line = "%s | %-24s | %-6s %s\n" % (time.strftime("%m-%d %H:%M"), _run_name()[:24], kind,
                                               " ".join(str(detail).split())[:200])
            with open(_ledger, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass

    # ── 記憶體（2026-10-02）：本行程 RSS／峰值 RSS／子行程（DataLoader worker）RSS 合計，讀 /proc，不需 psutil ──
    #   ⚠ 子行程是 fork 出來的，共用頁面在 RSS 裡會重複計算 => 合計是上界。
    def _rss_mib():
        me = os.getpid(); rss = hwm = kids = 0.0
        try:
            for ln in open("/proc/self/status"):
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1]) / 1024
                elif ln.startswith("VmHWM:"):
                    hwm = int(ln.split()[1]) / 1024
        except Exception:
            pass
        try:
            for d in os.listdir("/proc"):
                if not d.isdigit():
                    continue
                try:
                    st = open(f"/proc/{d}/stat").read()
                    if int(st.rsplit(")", 1)[1].split()[1]) != me:
                        continue
                    for ln in open(f"/proc/{d}/status"):
                        if ln.startswith("VmRSS:"):
                            kids += int(ln.split()[1]) / 1024
                            break
                except Exception:
                    continue
        except Exception:
            pass
        return rss, hwm, kids

    # ── 真實迴圈逐段計時（2026-10-02）：**只在設了 CITYGS_STEPPROF_OUT 時作用**（專用的短跑次，不用在正式訓練）──
    #   對應我方 `_StepProfiler`（CITYGS_STEP_PROFILE）：每個標記點 synchronize、量 wall ms 與段內峰值配置。
    #   官方原始碼一行不改：在 import 時包住 forward／loss／backward／density after_backward／Adam.step／trim。
    #   ⚠ 會 reset 峰值統計 => 這種跑次的峰值欄不可當資源數字；同步會消掉 CPU/GPU 重疊 => 段落和 > 真實每步。
    _sp = os.environ.get("CITYGS_STEPPROF_OUT")
    _SP = {"on": False, "acc": {}, "steps": 0, "tot": 0.0, "prev": None, "depth": {},
           "warm": int(os.environ.get("CITYGS_STEPPROF_WARMUP", "20"))}

    def _sp_wrap(fn, seg, only_training=False):
        def w(*a, **kw):
            if (not _SP["on"]) or _SP["depth"].get(seg, 0) > 0 or \
                    (only_training and a and not getattr(a[0], "training", True)):
                return fn(*a, **kw)
            c = sys.modules["torch"].cuda
            _SP["depth"][seg] = 1
            try:
                c.synchronize(); c.reset_peak_memory_stats(); t0 = time.perf_counter()
                r = fn(*a, **kw)
                c.synchronize(); dt = (time.perf_counter() - t0) * 1e3
                e = _SP["acc"].setdefault(seg, [0.0, 0, 0.0, 0.0])
                e[0] += dt; e[1] += 1
                e[2] = max(e[2], c.max_memory_allocated() / 2 ** 20); e[3] = c.memory_reserved() / 2 ** 20
                return r
            finally:
                _SP["depth"][seg] = 0
        w._citygs_wrapped = True
        return w

    def _sp_patch_methods(mod, names_to_seg, only_training=()):
        for nm in dir(mod):
            cls = getattr(mod, nm)
            if not isinstance(cls, type) or getattr(cls, "__module__", None) != mod.__name__:
                continue
            for meth, seg in names_to_seg.items():
                f = cls.__dict__.get(meth)
                if f is not None and not getattr(f, "_citygs_wrapped", False):
                    setattr(cls, meth, _sp_wrap(f, seg, meth in only_training))

    def _sp_tick(trainer):
        # 每步的真實總時間＝相鄰兩次 on_train_batch_start 的間隔（含 trim 與迴圈外）
        if not _sp:
            return
        c = sys.modules["torch"].cuda
        c.synchronize()
        now = time.perf_counter()
        if _SP["on"] and _SP["prev"] is not None:
            _SP["tot"] += (now - _SP["prev"]) * 1e3; _SP["steps"] += 1
        _SP["on"] = trainer.global_step >= _SP["warm"]
        _SP["prev"] = now

    def _sp_report(n):
        if not _sp or not _SP["steps"]:
            return
        S = _SP["steps"]; meas = 0.0
        lines = [f"# 官方程式碼真實迴圈逐段計時（每標記點 synchronize；暖機 {_SP['warm']} 步）  N={n:,}  run={_run_name()}",
                 f"# 計時步數 {S}  真實每步（相鄰 batch_start 間隔，含同步開銷） {_SP['tot'] / S:.2f} ms",
                 "segment\tms_per_step\tcalls_per_step\tms_per_call\tseg_peak_alloc_MiB\treserved_MiB_after"]
        for seg, (ms, cnt, pk, rs) in sorted(_SP["acc"].items()):
            lines.append(f"{seg}\t{ms / S:.2f}\t{cnt / S:.3f}\t{ms / max(cnt, 1):.2f}\t{pk:.0f}\t{rs:.0f}")
            meas += ms
        lines.append(f"0 其他（迴圈外＋未包到的）\t{(_SP['tot'] - meas) / S:.2f}\t-\t-\t-\t-")
        try:
            with open(_sp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    # ── churn：包住 density controller 的函式（只數數量）──────────────────────────
    def _n_of(gm):
        try:
            return int(gm.get_xyz.shape[0])
        except Exception:
            try:
                return int(gm.n_gaussians)
            except Exception:
                return -1

    def _event(phase, added, removed, gm):
        t = _S["tot"].setdefault(phase, [0, 0, 0])
        t[0] += added; t[1] += removed; t[2] += 1
        _S["last_ev"] = f"step {_S['step']:,} {phase} +{added:,} -{removed:,} => N {_n_of(gm):,}"
        _append(_churn, "global_step\tphase\tadded\tremoved\tN_after",
                f"{_S['step']}\t{phase}\t{added}\t{removed}\t{_n_of(gm)}")

    def _wrap_phase(fn, phase):
        def w(self, *a, **kw):
            _S["phase"].append(phase)
            try:
                return fn(self, *a, **kw)
            finally:
                _S["phase"].pop()
        w._citygs_wrapped = True
        return w

    def _wrap_postfix(fn):
        def w(self, new_properties, gaussian_model, *a, **kw):
            try:
                k = int(next(iter(new_properties.values())).shape[0])
            except Exception:
                k = -1
            r = fn(self, new_properties, gaussian_model, *a, **kw)
            _event(_S["phase"][-1] if _S["phase"] else "other_add", k, 0, gaussian_model)
            return r
        w._citygs_wrapped = True
        return w

    def _wrap_prune(fn):
        def w(self, mask, gaussian_model, *a, **kw):
            try:
                k = int(mask.sum().item())
            except Exception:
                k = -1
            r = fn(self, mask, gaussian_model, *a, **kw)
            _event(_S["phase"][-1] if _S["phase"] else "trim", 0, k, gaussian_model)
            return r
        w._citygs_wrapped = True
        return w

    def _wrap_reset(fn):
        def w(self, gaussian_model, *a, **kw):
            r = fn(self, gaussian_model, *a, **kw)
            _event("reset_opacity", 0, 0, gaussian_model)
            return r
        w._citygs_wrapped = True
        return w

    _WRAP = {"_densify_and_clone": lambda f: _wrap_phase(f, "clone"),
             "_densify_and_split": lambda f: _wrap_phase(f, "split"),
             "_densify_and_prune": lambda f: _wrap_phase(f, "cull"),
             "_densification_postfix": _wrap_postfix,
             "_prune_points": _wrap_prune,
             "_reset_opacities": _wrap_reset}

    def _patch_density(mod):
        n = 0
        for obj in list(vars(mod).values()):
            if not isinstance(obj, type) or obj.__module__ != mod.__name__:
                continue
            for name, mk in _WRAP.items():
                f = obj.__dict__.get(name)
                if f is not None and not getattr(f, "_citygs_wrapped", False):
                    setattr(obj, name, mk(f)); n += 1
        if n:
            print(f"[trainlog] ✅ churn 計數掛上 {mod.__name__}（{n} 個函式）", file=sys.stderr)

    # ── 訓練曲線＋台帳：唯讀 callback ────────────────────────────────────────────
    def _make_cb(Callback):
        class _TrainLog(Callback):
            def __init__(self):
                self.k = self.kl = 0
                self.t0 = self.tl = time.time()
                self.last = {}

            def _row(self, trainer, pl_module, tag=""):
                t = sys.modules["torch"]
                now = time.time()
                its = (self.k - self.kl) / max(now - self.tl, 1e-9)
                self.kl, self.tl = self.k, now
                n = _n_of(pl_module.gaussian_model) if hasattr(pl_module, "gaussian_model") else -1
                m = [0.0] * 4
                try:
                    c = t.cuda
                    m = [c.memory_allocated() / 2 ** 20, c.memory_reserved() / 2 ** 20,
                         c.max_memory_allocated() / 2 ** 20, c.max_memory_reserved() / 2 ** 20]
                except Exception:
                    pass
                self.last = dict(step=trainer.global_step, n=n, its=its, a=m[0], r=m[1], pa=m[2], pr=m[3], wall=now - self.t0)
                rs, hw, kd = _rss_mib()
                _append(_tl, "batch\tglobal_step\twall_s\tit_s\tN\talloc_MiB\treserved_MiB\tpeak_alloc_MiB\tpeak_reserved_MiB\ttag"
                             "\trss_MiB\trss_peak_MiB\tworkers_rss_MiB",
                        f"{self.k}\t{trainer.global_step}\t{now - self.t0:.1f}\t{its:.3f}\t{n}\t"
                        + "\t".join(f"{v:.0f}" for v in m) + f"\t{tag}\t{rs:.0f}\t{hw:.0f}\t{kd:.0f}")

            def _status(self, trainer, phase):
                L = self.last
                ms = trainer.max_steps or 0
                st = L.get("step", 0)
                wall = L.get("wall", 0.0)
                avg = st / wall if wall > 0 else 0.0
                eta = (ms - st) / avg if avg > 0 and ms else 0
                lines = [
                    f"# 官方參考線現況（{_run_name()}）  更新 {time.strftime('%m-%d %H:%M:%S')}",
                    f"狀態     {phase}",
                    f"步數     {st:,} / {ms:,}（{100 * st / ms:.1f}%）" if ms else f"步數     {st:,}",
                    f"N        {L.get('n', 0):,}",
                    f"速度     區間 {L.get('its', 0):.2f} it/s ／ 全程平均 {avg:.2f} it/s",
                    f"經過     {wall / 3600:.2f} h   預估剩餘 {eta / 3600:.2f} h",
                    f"VRAM     目前 {L.get('a', 0) / 1024:.2f}G 配置／{L.get('r', 0) / 1024:.2f}G 保留   "
                    f"累計峰值 {L.get('pa', 0) / 1024:.2f}G／{L.get('pr', 0) / 1024:.2f}G",
                ]
                try:
                    ps = [f"{k}={float(v):.4f}" for k, v in trainer.callback_metrics.items()
                          if any(x in k.lower() for x in ("psnr", "ssim", "lpips"))]
                    if ps:
                        lines.append("val      " + " ".join(ps))
                except Exception:
                    pass
                tot = _S["tot"]
                if tot:
                    lines.append("churn    " + " ".join(f"{p}: +{a:,}/-{r:,}（{c} 次）" for p, (a, r, c) in tot.items()))
                    lines.append("最近事件 " + _S["last_ev"])
                txt = "\n".join(lines) + "\n"
                for d in {getattr(trainer, "default_root_dir", None), None}:
                    path = os.path.join(d, "train_status.txt") if d else _status_extra
                    if not path:
                        continue
                    try:
                        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                        tmp = path + ".tmp"
                        with open(tmp, "w", encoding="utf-8") as f:
                            f.write(txt)
                        os.replace(tmp, path)
                    except Exception:
                        pass

            def _snap(self, trainer):
                L = self.last
                s = (f"step={L.get('step', 0):,}/{trainer.max_steps:,} N={L.get('n', 0) / 1e6:.2f}M "
                     f"{L.get('its', 0):.2f}it/s 峰值實佔{L.get('pa', 0) / 1024:.2f}G/保留{L.get('pr', 0) / 1024:.2f}G "
                     f"牆鐘{L.get('wall', 0) / 3600:.2f}h")
                try:
                    for k, v in trainer.callback_metrics.items():
                        if "psnr" in k.lower():
                            s += f" {k}={float(v):.3f}"
                            break
                except Exception:
                    pass
                tot = _S["tot"]
                if tot:
                    s += " churn[" + " ".join(f"{p}+{a}-{r}" for p, (a, r, _) in tot.items() if a or r) + "]"
                return s

            def setup(self, trainer, pl_module, stage):
                if stage == "fit":
                    self._status(trainer, "準備中：建立資料集／快取影像（尚未開始訓練）")

            def on_train_start(self, trainer, pl_module):
                self.t0 = self.tl = time.time()
                self._row(trainer, pl_module, "START")
                self._status(trainer, "訓練中")
                _progress("START", f"max_steps={trainer.max_steps:,} N0={self.last.get('n', 0):,} "
                                   f"env={os.environ.get('CONDA_DEFAULT_ENV', '?')} out={os.getcwd()}")

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                _S["step"] = trainer.global_step
                _sp_tick(trainer)
                if _sp and _SP["steps"] and _SP["steps"] % 200 == 0:
                    _sp_report(_n_of(pl_module.gaussian_model) if hasattr(pl_module, "gaussian_model") else -1)

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.k += 1
                if self.k % _every == 0:
                    self._row(trainer, pl_module)
                    self._status(trainer, "訓練中")

            def on_train_end(self, trainer, pl_module):
                _sp_report(_n_of(pl_module.gaussian_model) if hasattr(pl_module, "gaussian_model") else -1)
                self._row(trainer, pl_module, "END")
                self._status(trainer, "✅ 完成")
                _progress("DONE", self._snap(trainer))

            def on_exception(self, trainer, pl_module, exception):
                self._row(trainer, pl_module, "DIED " + repr(exception)[:300].replace("\t", " ").replace("\n", " "))
                self._status(trainer, "⛔ DIED " + " ".join(str(exception).split())[:200])
                _progress("DIED", f"{type(exception).__name__}: {' '.join(str(exception).split())[:90]} | {self._snap(trainer)}")

        return _TrainLog

    def _patch_trainer(mod):
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

    _TARGETS = {"lightning.pytorch.trainer.connectors.callback_connector": _patch_trainer,
                "internal.density_controllers.vanilla_density_controller": _patch_density,
                "internal.density_controllers.citygsv2_density_controller": _patch_density}
    if _sp:
        def _sp_gs(mod):
            _sp_patch_methods(mod, {"forward": "3 forward（光柵化）"}, only_training=("forward",))
            # manual_backward 是從 LightningModule 繼承的（不在子類的 __dict__）=> 直接在 GaussianSplatting 上覆寫一層
            cls = getattr(mod, "GaussianSplatting", None)
            if cls is not None:
                f = getattr(cls, "manual_backward", None)
                if f is not None and not getattr(f, "_citygs_wrapped", False):
                    cls.manual_backward = _sp_wrap(f, "6 backward")

        def _sp_metric(mod):
            _sp_patch_methods(mod, {"get_train_metrics": "4 loss（含正則）"})

        def _sp_dens(mod):
            _sp_patch_methods(mod, {"after_backward": "7 density_controller.after_backward"})

        def _sp_rend(mod):
            _sp_patch_methods(mod, {"before_training_step": "1 起始 trim（before_training_step）",
                                    "after_training_step": "11 週期 trim（after_training_step）"})

        def _sp_torch_adam():
            import torch as _t
            for cls in (_t.optim.Adam,):
                f = cls.__dict__.get("step")
                if f is not None and not getattr(f, "_citygs_wrapped", False):
                    cls.step = _sp_wrap(f, "9 optimizer.step")

        def _chain(*fs):
            def g(m):
                for f in fs:
                    f(m)
            return g

        _orig_trainer_patch = _TARGETS["lightning.pytorch.trainer.connectors.callback_connector"]
        _TARGETS["lightning.pytorch.trainer.connectors.callback_connector"] = _chain(_orig_trainer_patch, lambda m: _sp_torch_adam())
        _TARGETS["internal.gaussian_splatting"] = _sp_gs
        for _mn in ("internal.metrics.citygsv2_metrics",):
            _TARGETS[_mn] = _sp_metric
        for _mn in ("internal.density_controllers.vanilla_density_controller",
                    "internal.density_controllers.citygsv2_density_controller"):
            _TARGETS[_mn] = _chain(_TARGETS[_mn], _sp_dens)
        _TARGETS["internal.renderers.sep_depth_trim_2dgs_renderer"] = _sp_rend
        print(f"[stepprof] ✅ 逐段計時開啟 -> {_sp}（暖機 {_SP['warm']} 步）", file=sys.stderr)

    class _Hook(importlib.abc.MetaPathFinder):
        def __init__(self):
            self.busy = set()

        def find_spec(self, name, path, target=None):
            if name not in _TARGETS or name in self.busy:
                return None
            self.busy.add(name)
            try:
                spec = None
                for f in sys.meta_path:
                    if f is self or not hasattr(f, "find_spec"):
                        continue
                    spec = f.find_spec(name, path, target)
                    if spec is not None:
                        break
            finally:
                self.busy.discard(name)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec
            _orig_exec = spec.loader.exec_module
            patch = _TARGETS[name]

            def exec_module(module):
                _orig_exec(module)
                try:
                    patch(module)
                except Exception as e:
                    print(f"[trainlog] ⚠⚠ {name} 掛不上：{e!r}", file=sys.stderr)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
