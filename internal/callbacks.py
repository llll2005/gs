import os
import sys
import math
import time
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm


class SaveCheckpoint(Callback):
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_rank != 0:
            return
        checkpoint_path = os.path.join(
            pl_module.hparams["output_path"],
            "checkpoints",
            "epoch={}-step={}.ckpt".format(trainer.current_epoch, trainer.global_step),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)


class SaveGaussian(Callback):
    def on_train_end(self, trainer, pl_module) -> None:
        # TODO: should save before densification
        pl_module.save_gaussians()


class KeepRunningIfWebViewerEnabled(Callback):
    def on_train_end(self, trainer, pl_module) -> None:
        if pl_module.web_viewer is None:
            return
        print("Training finished! Web viewer is still running. Press `Ctrl+C` to exist.")
        while True:
            pl_module.web_viewer.is_training_paused = True
            pl_module.web_viewer.process_all_render_requests(pl_module.gaussian_model, pl_module.renderer, pl_module._fixed_background_color())


class StopImageSavingThreads(Callback):
    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        alive_threads = pl_module.image_saving_threads

        while len(alive_threads) > 0:
            # send messages to terminate threads
            while True:
                try:
                    pl_module.image_queue.put(None, block=False)
                except:
                    break

            # check whether any threads are alive
            still_alive_threads = []
            for thread in alive_threads:
                if thread.is_alive() is True:
                    still_alive_threads.append(thread)
            alive_threads = still_alive_threads


class ProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position + 1)
        self.on_epoch_metrics = {}

    def get_metrics(self, trainer, model):
        # only return latest logged metrics
        items = trainer._logger_connector.metrics["pbar"]
        return items

    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        self.max_epochs = trainer.max_epochs
        if self.max_epochs < 0:
            self.max_epochs = math.ceil(trainer.max_steps / self.total_train_batches)

        self.epoch_progress_bar = Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position) - 1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            total=self.max_epochs,
        )
        self.epoch_progress_bar.update(trainer.current_epoch)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.on_epoch_metrics.update(self.get_metrics(trainer, pl_module))
        self.epoch_progress_bar.set_postfix(self.on_epoch_metrics)
        self.epoch_progress_bar.update()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self.on_epoch_metrics.update(self.get_metrics(trainer, pl_module))


class ValidateOnTrainEnd(Callback):
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_last_batch is False or trainer.current_epoch % trainer.check_val_every_n_epoch != 0:
            trainer.validating = True
            trainer._evaluation_loop.run()
            trainer.validating = False


class StopDataLoaderCacheThread(Callback):
    def _stop_thread(self, data_loader):
        import queue
        if data_loader.cache_thread is not None:
            data_loader.stop_caching = True
            try:
                data_loader.cache_output_queue.get(block=False)
            except queue.Empty:
                pass
            data_loader.cache_thread.join()

    def on_train_end(self, trainer, pl_module) -> None:
        self._stop_thread(trainer.train_dataloader)


class TrainConsole(ProgressBar):
    """統一訓練輸出（預設進度條，取代 tqdm）。TTY+rich → 底部固定 rich 面板（最近 val/進度/VRAM，>90% 紅字）；
    非 TTY / rich 不可用 / live=false → **靜音 tqdm**，改每 plain_every_n_steps 步印一次精簡快照
    （2026-09-13 改；原本退回 tqdm，但非 TTY 沒有原地更新 => 每次 refresh 都是新的一行，
     lab 的 slot log 曾長到 267,581 行）。
    兩種模式都寫 outputs/<run>/train_status.txt（可 cat/貼 AI）+ 收 RTG 事件。任何 rich 失敗都不影響訓練。"""

    def __init__(self, refresh_rate: int = 1, process_position: int = 0,
                 live: bool = True, status_file: bool = True,
                 plain_every_n_steps: int = 0, rich_refresh_per_second: float = 4.0):
        super().__init__(refresh_rate, process_position)
        from internal.utils.train_console import get_state
        self.live_enabled = live
        self.status_file = status_file
        self.plain_every_n_steps = plain_every_n_steps
        self.rich_refresh_per_second = rich_refresh_per_second
        self._state = get_state()
        self._live = None
        self._rich_ok = False
        self._t_last = None
        self._step_last = 0
        self._last_status_write = 0.0
        self._last_plain_print = 0

    def _build_header(self, trainer, pl_module):
        import torch
        lines = []
        try:
            dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            tot = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        except Exception:
            dev, tot = "?", 0
        hp = pl_module.hparams
        sh = getattr(getattr(pl_module, "gaussian_model", None), "max_sh_degree", "?")
        init = hp.get("initializer", None)
        if init is not None:
            p = str(getattr(init, "path", "") or "")
            init_s = type(init).__name__ + "(" + (os.path.basename(p) or "-") + ")"
        else:
            ifrom = hp.get("initialize_from")
            init_s = f"initialize_from={os.path.basename(str(ifrom)) if ifrom else None}"
        dens = hp.get("density", None)
        dens_s = type(dens).__name__ if dens is not None else "?"
        extra = ""
        cap = getattr(dens, "cap_max", 0) if dens is not None else 0
        if cap:
            band = getattr(dens, "soft_cap_band", 0)
            soft = getattr(dens, "soft_cap_enabled", False)
            extra = f" cap {cap/1e6:.2f}M" + (f" soft[{(cap-band)/1e6:.2f}M,{cap/1e6:.2f}M]" if soft else "")
        try:
            n_init = pl_module.gaussian_model.n_gaussians
        except Exception:
            n_init = 0
        lines.append(f"dev {dev} {tot:.1f}G | sh{sh} | init {init_s}")
        lines.append(f"density {dens_s}{extra} | init顆數 {n_init:,}")
        return lines

    def setup(self, trainer, pl_module, stage):
        # Capture the run identity once. Without this every ledger line reads "CityGaussian" and
        # only START is attributable (it happens to carry `out=<path>` in its detail); DONE and
        # DIED carry only a step/VRAM snapshot, so pairing them with a run relies on ordering and
        # breaks as soon as two runs interleave.
        try:
            parts = [q for q in str(pl_module.hparams["output_path"]).rstrip("/").split(os.sep) if q]
            if len(parts) >= 3 and parts[-2] == "blocks":
                self._ledger_name = f"{parts[-3]}/{parts[-1]}"   # <run>/block_N
            elif parts:
                self._ledger_name = parts[-1]
        except Exception:
            pass
        super().setup(trainer, pl_module, stage)
        if stage != "fit":
            return
        name = os.path.basename(str(trainer.default_root_dir or "run"))
        st = self._state
        st.set_header(name, self._build_header(trainer, pl_module))
        if self.status_file:
            base = pl_module.hparams.get("output_path") or trainer.default_root_dir or "."
            st.status_path = os.path.join(base, "train_status.txt")
            # 立刻寫一份：從這裡到第一個訓練步之間會經過「建資料集 + 快取全部影像」，
            # 那段時間以前完全沒有輸出（使用者 2026-09-12：「沒這個我都不知道訓練到哪了」）。
            st.set_footer(step=0, max_steps=trainer.max_steps,
                          phase="準備中：建立資料集 / 快取影像（尚未開始訓練）")
            st.write_status_file()
        try:
            h = pl_module.hparams
            cap = None
            d = getattr(getattr(pl_module, "density", None), "config", None)
            if d is not None:
                cap = getattr(d, "cap_max", None)
            m = getattr(getattr(pl_module, "metric", None), "config", None)
            oreg = getattr(m, "opacity_reg", None) if m is not None else None
            det = f"max_steps={trainer.max_steps:,}"
            if cap:
                det += f" cap={cap / 1e6:.2f}M"
            if oreg is not None:
                det += f" opacity_reg={oreg}"
            det += f" out={base}"
        except Exception:
            det = ""
        self._progress("START", det)

    def _try_start_rich(self):
        if not self.live_enabled:
            return False
        if not (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
            return False
        try:
            from rich.live import Live
            from rich.console import Console
            self._live = Live(self._state.render_rich(), console=Console(),
                              refresh_per_second=self.rich_refresh_per_second, transient=False)
            self._live.start()
            return True
        except Exception:
            self._live = None
            return False

    def on_train_start(self, trainer, pl_module):
        # 此時 gaussian_model 已被 initializer 灌點 → 重建 header 才有正確 init 顆數（修 setup 期顯示 0）
        name = os.path.basename(str(trainer.default_root_dir or "run"))
        self._state.set_header(name, self._build_header(trainer, pl_module))
        self._state.set_footer(phase=None)        # 準備階段結束 => 讓底欄回到正常進度顯示
        self._rich_ok = self._try_start_rich()
        self._state.active = self._rich_ok        # 只在 rich 模式緩衝事件；fallback 保留原 print 行為
        if self._rich_ok:
            self.disable()                         # 靜音 tqdm（所有 bar no-op），改由 rich 驅動
        else:
            # ⚠⚠ 2026-09-13：非 TTY **也要**靜音 tqdm。
            #   原本只在 rich 模式 disable，非 TTY 就退回 tqdm —— 但非 TTY 正是最不該用進度條的
            #   場合：沒有 TTY 就沒有原地更新，每次 refresh 都變成新的一行。
            #   實測 lab 的 logs/runner.slot2.log 長到 **267,581 行**，幾乎全是進度條，
            #   要找一段錯誤得掃過幾十萬行，而且透過 API 讀很慢。
            #   資訊沒有損失：`train_status.txt` 每 2 秒就寫一次完整快照。
            #   ⇒ 關掉 tqdm，改成每 N 步印一次同一份精簡快照（沿用既有的 plain 路徑）。
            self.disable()
            if self.plain_every_n_steps <= 0:
                self.plain_every_n_steps = 500
        super().on_train_start(trainer, pl_module)
        self._t_last = time.time()
        self._step_last = trainer.global_step

    def _stop_live(self):
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        self._state.active = False


    # ---- run-progress ledger -------------------------------------------------
    # One structured line per run lifecycle event, appended to logs/quad_progress.log.
    # Emitted from here rather than from the wrapper scripts: the scripts used to grep it back
    # out of the training log, which dragged tqdm bars and ANSI escapes into the ledger, and
    # every script had drifted to its own column widths. The callback already holds the numbers.
    PROGRESS_LOG = "logs/quad_progress.log"
    PROGRESS_FMT = "%s | %-24s | %-6s %s\n"

    def _progress(self, kind: str, detail: str) -> None:
        try:
            import time as _t
            os.makedirs(os.path.dirname(self.PROGRESS_LOG), exist_ok=True)
            line = self.PROGRESS_FMT % (_t.strftime("%m-%d %H:%M"), self._run_name()[:24], kind,
                                        " ".join(str(detail).split())[:200])
            with open(self.PROGRESS_LOG, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass                                   # a ledger must never take the run down

    def _run_name(self) -> str:
        """Identify the run in the ledger.

        `self._state.run_name` is never populated, so every line used to read "CityGaussian" and
        only START could be attributed at all -- it happens to carry `out=<path>` in its detail.
        DONE and DIED carry only a step/VRAM snapshot, so pairing them with a run relied on
        ordering, which breaks the moment two runs interleave.

        `output_path` is `outputs/<name>/blocks/block_<id>`, so the two trailing components give
        `<name>/block_N` -- unique per arm and short enough for the column.
        """
        return (getattr(self._state, "run_name", None)
                or getattr(self, "_ledger_name", None) or "run")

    def _progress_snapshot(self) -> str:
        """step/N/speed/VRAM straight from the live footer — no parsing of anything."""
        try:
            f = dict(self._state.footer)
        except Exception:
            return ""
        bits = []
        if f.get("max_steps"):
            bits.append(f"step={f.get('step', 0):,}/{f['max_steps']:,}")
        elif f.get("step"):
            bits.append(f"step={f['step']:,}")
        if "n_gauss" in f:
            bits.append(f"N={f['n_gauss'] / 1e6:.2f}M")
        if "its" in f:
            bits.append(f"{f['its']:.2f}it/s")
        if "vram_used" in f:
            _a = f.get("vram_alloc")
            bits.append(f"VRAM={f['vram_used']:.2f}/{f.get('vram_total', 0):.1f}G"
                        + (f"(峰值實佔{_a:.2f})" if _a else ""))
        try:
            if self._state.vals:
                vstep, vm = self._state.vals[-1]
                if vm.get("psnr") is not None:
                    bits.append(f"PSNR={vm['psnr']:.3f}@{vstep:,}")
        except Exception:
            pass
        return " ".join(bits)

    def on_train_end(self, trainer, pl_module):
        try:
            super().on_train_end(trainer, pl_module)
        except Exception:
            pass
        self._state.write_status_file()
        self._progress("DONE", self._progress_snapshot())
        self._stop_live()

    def on_exception(self, trainer, pl_module, exception):
        self._progress("DIED", f"{type(exception).__name__}: "
                               f"{' '.join(str(exception).split())[:90]} | {self._progress_snapshot()}")
        self._stop_live()                          # 先停 Live，讓 traceback 完整顯示
        try:
            super().on_exception(trainer, pl_module, exception)
        except Exception:
            pass

    def teardown(self, trainer, pl_module, stage):
        self._stop_live()
        try:
            super().teardown(trainer, pl_module, stage)
        except Exception:
            pass

    def _vram(self):
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        # 第三個值 = **峰值** allocated（`max_memory_allocated`），前者是 reserved。
        # ⚠⚠ 2026-09-05 修：原本用 `memory_allocated()`（**瞬時**）。footer 在步與步之間更新，
        #    暫存已釋放 ⇒ 它顯示的是**常駐量**（參數 0.506 + 梯度 0.506 + Adam 1.011 ≈ 2.02 GB），
        #    於是看起來「5.53 裡有 3.3 GB 在浪費」—— **那正是 §11.86 已撤回的錯誤結論**
        #    （真實碎片只有 0.933 GB，且 2.80M 顆直接 OOM）。
        #    峰值才是決定 OOM 的量：`tools/vram_scale.py` 實測 N=2.34M 時 peak alloc **3.736 GB**。
        #    `max_memory_allocated` 不重置 ⇒ 是整個跑次的峰值，正是要看的。
        return (torch.cuda.memory_reserved() / 1e9,
                torch.cuda.get_device_properties(0).total_memory / 1e9,
                torch.cuda.max_memory_allocated() / 1e9)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # tqdm 記帳（rich 模式 no-op）
        step = trainer.global_step
        now = time.time()
        its = 0.0
        if self._t_last is not None and now > self._t_last and step > self._step_last:
            its = (step - self._step_last) / (now - self._t_last)
        if now - (self._t_last or now) >= 1.0:
            self._t_last, self._step_last = now, step
        loss = trainer.callback_metrics.get("train/loss")
        used, tot, alloc = self._vram()
        try:
            n = pl_module.gaussian_model.n_gaussians
        except Exception:
            n = 0
        eta = "-"
        if its > 0 and trainer.max_steps:
            rem = max(0, trainer.max_steps - step) / its
            eta = f"{int(rem//3600)}h{int((rem % 3600)//60)}m"
        self._state.set_footer(
            step=step, max_steps=trainer.max_steps, epoch=trainer.current_epoch,
            loss=float(loss) if loss is not None else None,
            n_gauss=n, vram_used=used, vram_total=tot, vram_alloc=alloc, its=its, eta=eta,
        )
        if self._rich_ok and self._live is not None:
            try:
                self._live.update(self._state.render_rich())
            except Exception:
                pass
        if self.status_file and now - self._last_status_write > 2.0:
            self._state.write_status_file()
            self._last_status_write = now
        if not self._rich_ok and self.plain_every_n_steps > 0 and step - self._last_plain_print >= self.plain_every_n_steps:
            self._last_plain_print = step
            print(self._state.render_plain())

    def on_validation_end(self, trainer, pl_module):
        try:
            super().on_validation_end(trainer, pl_module)
        except Exception:
            pass
        cm = trainer.callback_metrics
        def g(k):
            v = cm.get(k)
            return float(v) if v is not None else 0.0
        # texratio is shown because PSNR misleads on this scene: b12 is half flat water, and on
        # 2026-08-07 `uniform_60k_b12` had PSNR fall 22.21 -> 21.93 at the same step that LPIPS and
        # texratio both improved -- blur is the MSE-optimal answer, so sharpening can cost PSNR.
        # 0 when the metric is absent (models other than CityGSV2Metrics do not log it).
        m = {"psnr": g("val/psnr"), "ssim": g("val/ssim"), "lpips": g("val/lpips"),
             "texratio": g("val/texratio")}
        step = trainer.global_step
        self._state.add_val(step, m)
        tex = f" tex{m['texratio']:.3f}" if m["texratio"] > 0 else ""
        self._state.event(f"val step{step:,}: psnr{m['psnr']:.2f} ssim{m['ssim']:.3f} "
                          f"lpips{m['lpips']:.3f}{tex}")
        if self.status_file:
            self._state.write_status_file()
