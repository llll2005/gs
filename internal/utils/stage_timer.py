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
        _mark_store(name, time.time())
        return
    now = time.time()
    t0 = _t0()
    d = now - (_LAST if _LAST is not None else t0)
    _LAST = now
    _mark_store(name, now)
    print(f"[階段] {time.strftime('%H:%M:%S', time.localtime(now))} +{now - t0:7.1f}s（+{d:6.1f}s）{name}", flush=True)


# ── ★ 11-14 每塊結束寫 run_summary.json（使用者：demo 模式不產生 log，但要有訓練總時間、顆數等寫論文要的資訊）──
#   mark() 的時間點＋record() 記的值，cli 結束時由 main.py 呼叫 write_summary() 寫到 <輸出目錄>/run_summary.json。
#   時間定義：startup_s＝行程開始 → 第一步開始；train_s＝第一步開始 → 最後一步結束（純訓練）；
#             save_exit_s＝最後一步結束 → cli 結束；python_total_s＝行程開始 → cli 結束。外層 shell 另外補 task_total_s。
_MARKS = {}
_REC = {}


def record(key: str, value) -> None:
    _REC[key] = value


def _mark_store(name: str, t: float) -> None:
    _MARKS.setdefault(name, t)


def write_summary() -> None:
    import json
    out = _REC.get("output_path")
    if not out:
        return
    t0 = _t0()
    m = {k: v - t0 for k, v in _MARKS.items()}
    fs = m.get("第一步開始（拿到第一個 batch）")
    last = max((v for k, v in m.items() if k.startswith("第 ") and k.endswith("步結束")), default=None)
    end = m.get("cli 結束（訓練與存檔完成，行程即將結束）")
    s = dict(_REC)
    s.update({
        "startup_s": round(fs, 1) if fs is not None else None,
        "train_s": round(last - fs, 1) if (fs is not None and last is not None) else None,
        "save_exit_s": round(end - last, 1) if (end is not None and last is not None) else None,
        "python_total_s": round(end, 1) if end is not None else None,
        "marks_s": {k: round(v, 1) for k, v in sorted(m.items(), key=lambda kv: kv[1])},
        "demo": os.environ.get("CITYGS_DEMO", "0") == "1",
    })
    try:
        with open(os.path.join(out, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        print(f"[run-summary] {os.path.join(out, 'run_summary.json')}｜啟動 {s['startup_s']} s、純訓練 {s['train_s']} s、"
              f"存檔結束 {s['save_exit_s']} s｜N {s.get('n_final')}｜峰值 {s.get('peak_alloc_gb')} GB", flush=True)
    except Exception as e:
        print(f"[run-summary] ⚠ 寫不進：{e}", flush=True)
