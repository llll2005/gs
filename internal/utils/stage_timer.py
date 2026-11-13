"""★ 2026-11-14 啟動／收尾時間拆解（使用者「啟動時間拆解 作」；紀錄/實驗分析/38 §4：每塊訓練外 3.5～5 分，其中啟動約 55 s）。

mark(name) 印一行：
    [階段] 16:18:42 +  23.4s（+  5.1s）<name>
  +X＝距這個 python 行程開始（psutil create_time，含直譯器啟動與 import）；（+Y）＝距上一個標記。
只印、不改任何計算；每個行程最多印幾十行。CITYGS_STAGE_TIMER=0 可關。
外層 shell（scripts/lab/_common.sh run_fit）用同樣格式印 conda run 之前／之後的時間（`[階段] HH:MM:SS shell …`）。
"""
import os
import time

_T0 = None
_LAST = None
_ON = os.environ.get("CITYGS_STAGE_TIMER", "1") != "0"


def _t0() -> float:
    global _T0
    if _T0 is None:
        try:
            import psutil
            _T0 = psutil.Process(os.getpid()).create_time()
        except Exception:
            _T0 = time.time()
    return _T0


def mark(name: str) -> None:
    global _LAST
    if not _ON:
        return
    now = time.time()
    t0 = _t0()
    d = now - (_LAST if _LAST is not None else t0)
    _LAST = now
    print(f"[階段] {time.strftime('%H:%M:%S', time.localtime(now))} +{now - t0:7.1f}s（+{d:6.1f}s）{name}", flush=True)
