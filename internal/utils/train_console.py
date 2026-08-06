"""
統一訓練輸出：一個全域狀態 + 兩種呈現。

目標（使用者需求）：
  - 最近 val 指標固定在顯眼處、進度更新不洗版、error 照常顯示；
  - 排版精簡、可整段貼給 AI、低 token、一目了然。

設計：
  - 一個 module 級單例 _STATE 收集：靜態表頭 / 最近事件 / 最近 val / 底欄即時指標。
  - `console_event(msg)`：RTG/renderer 等呼叫它推事件。**若沒有 Live callback 接管（active=False）
    就 fallback 成正常 print** → 沒開這功能時行為完全不變（向後相容）。
  - `render_plain()`：產生精簡純文字快照（寫進 outputs/<name>/train_status.txt，可 cat / 貼 AI）。
  - `render_rich()`：產生 rich 可渲染物件（給 TrainConsole callback 的 Live 固定底欄用）。
"""
from collections import deque
import threading
import time


class _TrainConsoleState:
    def __init__(self):
        self.run_name = ""
        self.header_lines = []                  # 靜態基礎資訊（dev / init / density 摘要…）
        self.events = deque(maxlen=6)           # (時刻, 訊息) 最近事件（RTG/warn…）
        self.vals = deque(maxlen=5)             # (step, {psnr,ssim,lpips}) 最近 val
        self.footer = {}                        # 即時：step/max/epoch/loss/n/vram/its/eta
        self.status_path = None
        self.active = False                     # 是否有 Live callback 接管顯示
        self._lock = threading.Lock()

    # ── 收集 ──────────────────────────────────────────────────────────────
    def set_header(self, run_name, lines):
        with self._lock:
            self.run_name = run_name
            self.header_lines = list(lines)

    def event(self, msg: str):
        with self._lock:
            self.events.append((time.strftime("%H:%M:%S"), msg))
            active = self.active
        if not active:
            print(msg)                          # fallback：沒開 console 就照舊 print

    def add_val(self, step: int, metrics: dict):
        with self._lock:
            self.vals.append((step, dict(metrics)))

    def set_footer(self, **kw):
        with self._lock:
            self.footer.update(kw)

    # ── 呈現：純文字（狀態檔 / 貼 AI）────────────────────────────────────
    def render_plain(self) -> str:
        with self._lock:
            f = dict(self.footer)
            header = list(self.header_lines)
            events = list(self.events)
            vals = list(self.vals)
            name = self.run_name
        L = []
        L.append(f"=== TRAIN {name} · {time.strftime('%H:%M:%S')} ===")
        L.extend(header)
        # progress 底欄
        step = f.get("step", 0); mx = f.get("max_steps", 0)
        pct = f" ({100*step/mx:.0f}%)" if mx else ""
        prog = f"step {step:,}/{mx:,}{pct} ep {f.get('epoch','-')}"
        bits = []
        if "n_gauss" in f: bits.append(f"{f['n_gauss']/1e6:.2f}M顆")
        if "loss" in f and f["loss"] is not None: bits.append(f"loss {f['loss']:.3f}")
        if "vram_used" in f:
            vu = f["vram_used"]; vt = f.get("vram_total", 0)
            frac = vu / vt if vt else 0
            mark = " !!HIGH" if frac >= 0.9 else (" !HI" if frac >= 0.8 else "")
            bits.append(f"VRAM {vu:.2f}/{vt:.1f}G{mark}")
        if "its" in f: bits.append(f"{f['its']:.2f} it/s")
        if "eta" in f: bits.append(f"ETA {f['eta']}")
        L.append("- progress " + "-" * 28)
        L.append(prog + (" │ " + " │ ".join(bits) if bits else ""))
        # 最近 val
        L.append("- latest val " + "-" * 26)
        if vals:
            for s, m in vals[-3:]:
                L.append(f"step{s:>7,}: PSNR {m.get('psnr',0):.2f} SSIM {m.get('ssim',0):.3f} LPIPS {m.get('lpips',0):.3f}")
        else:
            L.append("(尚無 val)")
        # 最近事件
        L.append("- recent events " + "-" * 23)
        for t, msg in events:
            L.append(f"{t} {msg}")
        return "\n".join(L) + "\n"

    def write_status_file(self):
        if not self.status_path:
            return
        try:
            with open(self.status_path, "w") as fh:
                fh.write(self.render_plain())
        except Exception:
            pass

    # ── 呈現：rich（終端固定底欄）────────────────────────────────────────
    def render_rich(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.console import Group
        from rich.text import Text

        with self._lock:
            f = dict(self.footer)
            header = list(self.header_lines)
            events = list(self.events)
            vals = list(self.vals)
            name = self.run_name

        head = Text("\n".join(header), style="dim")

        ev = Table.grid(padding=(0, 1))
        ev.add_column(style="cyan", no_wrap=True)
        ev.add_column()
        for t, msg in events:
            ev.add_row(t, msg)

        # footer：最近 val（醒目）+ 進度
        step = f.get("step", 0); mx = f.get("max_steps", 0)
        pct = f"{100*step/mx:.0f}%" if mx else "-"
        vline = "—"
        if vals:
            s, m = vals[-1]
            vline = f"[bold green]PSNR {m.get('psnr',0):.2f}[/]  SSIM {m.get('ssim',0):.3f}  LPIPS {m.get('lpips',0):.3f}  [dim]@step {s:,}[/]"
        vu = f.get("vram_used", 0); vt = f.get("vram_total", 0)
        frac = vu / vt if vt else 0
        if frac >= 0.9:
            vram_s = f"[bold red]VRAM {vu:.2f}/{vt:.1f}G !![/]"       # 逼近上限：紅字示警
        elif frac >= 0.8:
            vram_s = f"[yellow]VRAM {vu:.2f}/{vt:.1f}G[/]"
        else:
            vram_s = f"VRAM {vu:.2f}/{vt:.1f}G"
        prog = (f"[bold]step {step:,}/{mx:,}[/] ({pct})  ep {f.get('epoch','-')}  │  "
                f"{f.get('n_gauss',0)/1e6:.2f}M顆  │  {vram_s}  │  "
                f"{f.get('its',0):.2f} it/s  │  ETA {f.get('eta','-')}")
        footer = Group(Text.from_markup(vline), Text.from_markup(prog))

        return Group(
            Panel(head, title=f"[b]{name}[/]", title_align="left", border_style="dim", padding=(0, 1)),
            Panel(ev, title="recent events", title_align="left", border_style="dim", padding=(0, 1)),
            Panel(footer, title="latest val · progress", title_align="left", border_style="green", padding=(0, 1)),
        )


_STATE = _TrainConsoleState()


def get_state() -> "_TrainConsoleState":
    return _STATE


def console_event(msg: str):
    """RTG / renderer 等推事件。沒有 Live 接管時 fallback 成 print（行為不變）。"""
    _STATE.event(msg)
