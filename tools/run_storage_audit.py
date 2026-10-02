"""儲存結構稽核（2026-10-03）：ckpt 裡存了什麼、一個跑次目錄由什麼組成。我方與官方都用同一支量。

ckpt 的部分**不載入張量**：torch.save 的格式是 zip（data.pkl＋data/<key> 原始位元組），這裡用自訂 Unpickler
只記下每個張量的形狀與 dtype => 2,400 萬顆的官方合併 ckpt 也只吃幾十 MB 記憶體（lab 與訓練共用 RAM）。

輸出（每個 ckpt）：
  檔案大小、顆數 N
  模型參數（逐屬性：位元組、每顆 float 數）、優化器狀態（exp_avg／exp_avg_sq 逐參數）、density controller 緩衝、其他
  每顆位元組：參數／Adam／合計
輸出（每個目錄）：依類別加總 —— 終點 ckpt、中間 ckpt、PLY、TensorBoard／lightning_logs、驗證／測試影像、其他

用法：python tools/run_storage_audit.py <ckpt 或跑次目錄> [...]
"""
import os
import pickle
import re
import sys
import zipfile
from collections import defaultdict

ELEM = {"FloatStorage": 4, "HalfStorage": 2, "BFloat16Storage": 2, "DoubleStorage": 8, "LongStorage": 8,
        "IntStorage": 4, "ShortStorage": 2, "CharStorage": 1, "ByteStorage": 1, "BoolStorage": 1}


class FakeStorage:
    def __init__(self, typename, key, numel):
        self.typename, self.key, self.numel = typename, key, numel


class FakeTensor:
    def __init__(self, storage, shape):
        self.storage, self.shape = storage, tuple(shape)

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self):
        return self.numel * ELEM.get(self.storage.typename, 4)


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__["_state"] = st

    def __call__(self, *a, **k):
        return _Dummy()


def _rebuild_tensor(storage, offset, size, stride, *a, **k):
    return FakeTensor(storage, size)


def _rebuild_parameter(data, *a, **k):
    return data


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if mod == "torch._utils" and name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
            return _rebuild_parameter
        if mod == "torch" and name.endswith("Storage"):
            return name
        if mod == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if mod.startswith("builtins") or mod in ("collections", "copyreg", "_codecs"):
            return super().find_class(mod, name)
        return _Dummy

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        typ = pid[1]
        typename = typ if isinstance(typ, str) else getattr(typ, "__name__", str(typ))
        if not isinstance(typename, str) or "Storage" not in typename:
            typename = "FloatStorage"
        return FakeStorage(typename, pid[2], pid[4])


def load_shapes(path):
    with zipfile.ZipFile(path) as z:
        pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        with z.open(pkl) as f:
            obj = _Unpickler(f).load()
        sizes = {n.rsplit("/", 1)[-1]: z.getinfo(n).file_size for n in z.namelist() if "/data/" in n}
    return obj, sizes


def human(b):
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return f"{b:.1f} {u}" if u != "B" else f"{b} B"
        b /= 1024


def audit_ckpt(path):
    obj, sizes = load_shapes(path)
    fs = os.path.getsize(path)
    sd = obj.get("state_dict", {}) if isinstance(obj, dict) else {}
    pre = "gaussian_model.gaussians."
    params = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre) and isinstance(v, FakeTensor)}
    N = 0
    for k in ("means", "xyz"):
        if k in params:
            N = params[k].shape[0]
    if not N and params:
        N = max(v.shape[0] for v in params.values() if v.shape)
    dens = {k: v for k, v in sd.items() if k.startswith("density_controller.") and isinstance(v, FakeTensor)}
    other_sd = {k: v for k, v in sd.items() if not k.startswith(pre) and not k.startswith("density_controller.")
                and isinstance(v, FakeTensor)}
    opt = defaultdict(int)
    opt_other = 0
    for o in obj.get("optimizer_states", []) if isinstance(obj, dict) else []:
        names = {}
        for g in o.get("param_groups", []):
            for pid in g.get("params", []):
                names[pid] = g.get("name", str(pid))
        for pid, st in o.get("state", {}).items():
            for kk, vv in st.items():
                if isinstance(vv, FakeTensor):
                    if kk in ("exp_avg", "exp_avg_sq"):
                        opt[(names.get(pid, str(pid)), kk)] += vv.nbytes
                    else:
                        opt_other += vv.nbytes
    pb = sum(v.nbytes for v in params.values())
    ob = sum(opt.values()) + opt_other
    db = sum(v.nbytes for v in dens.values())
    xb = sum(v.nbytes for v in other_sd.values())
    tensor_total = sum(sizes.values())
    print(f"\n════ {path}")
    print(f"  檔案 {human(fs)}   N = {N:,}   張量合計 {human(tensor_total)}（其餘 {human(fs - tensor_total)} 是 pickle／超參數／zip 開銷）")
    print(f"  {'項目':<34}{'大小':>12}{'佔檔案':>8}{'每顆 B':>10}")
    rows = [("模型參數", pb), ("優化器狀態（Adam 兩個動量等）", ob), ("density controller 緩衝", db), ("state_dict 其他", xb)]
    for nm, b in rows:
        print(f"  {nm:<34}{human(b):>12}{100 * b / max(fs, 1):>7.1f}%{(b / N if N else 0):>10.1f}")
    print(f"  ── 參數逐屬性（每顆 float 數＝每顆位元組/4）")
    for k, v in sorted(params.items(), key=lambda kv: -kv[1].nbytes):
        per = v.nbytes / N if N else 0
        print(f"     {k:<20}{human(v.nbytes):>12}{100 * v.nbytes / max(pb, 1):>7.1f}%  {per / 4:>6.1f} float/顆  shape={v.shape}")
    if opt:
        print(f"  ── 優化器狀態逐參數")
        for (nm, kk), b in sorted(opt.items(), key=lambda kv: -kv[1]):
            print(f"     {nm:<14}{kk:<12}{human(b):>12}")
    if dens:
        print(f"  ── density controller 緩衝")
        for k, v in sorted(dens.items(), key=lambda kv: -kv[1].nbytes):
            print(f"     {k:<44}{human(v.nbytes):>12}  shape={v.shape}")
    return fs


def classify(rel):
    b = os.path.basename(rel)
    if b.endswith(".ckpt"):
        return "ckpt"
    if b.endswith(".ply"):
        return "PLY"
    if "events.out.tfevents" in b or "/lightning_logs/" in "/" + rel or "/tensorboard/" in "/" + rel:
        return "TensorBoard／lightning_logs"
    if re.search(r"\.(png|jpg|jpeg)$", b, re.I):
        return "影像（val／test 輸出）"
    if b.endswith((".txt", ".csv", ".log", ".json", ".yaml", ".tsv")):
        return "文字紀錄（txt／csv／yaml…）"
    return "其他"


def audit_dir(path):
    files = []
    for root, dirs, fs in os.walk(path):
        dirs[:] = [d for d in dirs if ".aborted_" not in d and d != "aborted_blocks"]
        for f in fs:
            p = os.path.join(root, f)
            try:
                files.append((os.path.relpath(p, path), os.path.getsize(p)))
            except OSError:
                pass
    ck = [(r, s) for r, s in files if r.endswith(".ckpt")]

    def step(r):
        m = re.search(r"step=(\d+)", r)
        return int(m.group(1)) if m else -1
    final = set()
    by_dir = defaultdict(list)
    for r, s in ck:
        by_dir[os.path.dirname(r)].append((step(r), -len(r), r))
    for d, lst in by_dir.items():
        final.add(max(lst)[2])
    cat = defaultdict(lambda: [0, 0])
    for r, s in files:
        c = classify(r)
        if c == "ckpt":
            c = "終點 ckpt" if r in final else "中間 ckpt"
        cat[c][0] += s; cat[c][1] += 1
    tot = sum(s for _, s in files)
    print(f"\n════ 目錄 {path}   合計 {human(tot)}（{len(files)} 個檔；不含 *.aborted_*）")
    for c, (s, n) in sorted(cat.items(), key=lambda kv: -kv[1][0]):
        print(f"  {c:<28}{human(s):>12}{100 * s / max(tot, 1):>7.1f}%  {n:>6} 個檔")
    return sorted(final)


def main():
    for p in sys.argv[1:]:
        if os.path.isdir(p):
            finals = audit_dir(p)
            for r in finals[:3]:
                audit_ckpt(os.path.join(p, r))
            if len(finals) > 3:
                print(f"  （另有 {len(finals) - 3} 個終點 ckpt 未逐項展開）")
        elif os.path.isfile(p):
            audit_ckpt(p)
        else:
            print(f"⛔ 找不到 {p}")


if __name__ == "__main__":
    main()
