"""★ 2026-11-14 demo 模式 A：COLMAP 檔快速讀取（使用者「做一個 demo 模式，開了就套用這些」；紀錄/實驗分析/38 啟動時間拆解）。

為什麼：每塊啟動時 dataparser 用純 Python 解析 images.bin（1.3 GB，含每張影像全部 2D 點）、metric 再解析一次 points3D.bin（383 萬點），
        lab 上合計約 25 s（本機 images 26 s、points3D 33 s）；dataparser 只用位姿／名稱／相機 id，2D 點從來沒用到。
做法（輸出與原本的讀法逐值相同，tests/colmap_fast_test.py 驗證）：
  read_images_binary_light  與 internal.utils.colmap.read_images_binary 相同，但跳過（seek）每張影像的 2D 點；
                            回傳的 Image 的 xys／point3D_ids 是空陣列 => 只能給不用 2D 點的地方用（dataparser）。
  points3D_arrays           points3D.bin 解析成陣列（xyz f8、rgb u1、error f8、track 的 image id 與每點的起訖），
                            快取到 <points3D.bin>.cache.npz（檔案大小＋修改時間不同就重算；寫入用暫存檔＋os.replace，多個行程同時寫也安全）。
  read_points3D_xyz_rgb_err 給 ColmapDataParser.read_points3D_binary 用的同介面版本（含 selected_image_ids 過濾，順序不變）。
啟用：環境變數 CITYGS_SFM_FAST=1（scripts/lab/_common.sh 的 demo 模式會設）。
"""
import os
import struct

import numpy as np

from internal.utils.colmap import Image, read_next_bytes


def enabled() -> bool:
    return os.environ.get("CITYGS_SFM_FAST", "0") == "1"


def read_images_binary_light(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = b""
            c = fid.read(1)
            while c != b"\x00":
                name += c
                c = fid.read(1)
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.seek(24 * num_points2D, 1)
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name.decode("utf-8"),
                                     xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), dtype=np.int64))
    return images


def _parse_points3D(path):
    with open(path, "rb") as f:
        buf = f.read()
    n = struct.unpack_from("<Q", buf, 0)[0]
    xyz = np.empty((n, 3), dtype=np.float64)
    rgb = np.empty((n, 3), dtype=np.uint8)
    err = np.empty((n,), dtype=np.float64)
    pid = np.empty((n,), dtype=np.int64)
    off = np.empty((n + 1,), dtype=np.int64)
    starts = np.empty((n,), dtype=np.int64)          # 每點 track 在檔案裡的位元組起點
    head = struct.Struct("<QdddBBBdQ")
    o = 8
    t = 0
    off[0] = 0
    for i in range(n):
        p, x, y, z, r, g, b, e, tl = head.unpack_from(buf, o)
        o += 51
        pid[i] = p
        xyz[i, 0] = x; xyz[i, 1] = y; xyz[i, 2] = z
        rgb[i, 0] = r; rgb[i, 1] = g; rgb[i, 2] = b
        err[i] = e
        starts[i] = o
        o += 8 * tl
        t += tl
        off[i + 1] = t
    # track：每個元素 (image_id int32, point2D_idx int32)，image_id 的位元組位置＝該點 track 起點 + 8 x 第幾個；
    #   位置不一定 4 對齊 => 用四個不複製的 int32 視圖（位移 0～3）各取一部分
    lens = np.diff(off)
    pos = np.arange(t, dtype=np.int64)
    pos -= np.repeat(off[:-1], lens)
    pos *= 8
    pos += np.repeat(starts, lens)
    img = np.empty((t,), dtype=np.int32)
    for k in range(4):
        m = (pos & 3) == k
        if m.any():
            v = np.frombuffer(buf, dtype="<i4", offset=k, count=(len(buf) - k) // 4)
            img[m] = v[(pos[m] - k) >> 2]
    return dict(pid=pid, xyz=xyz, rgb=rgb, err=err, off=off, img=img)


def points3D_arrays(path):
    st = os.stat(path)
    cache = path + ".cache.npz"
    key = np.array([st.st_size, st.st_mtime_ns], dtype=np.int64)
    if os.path.exists(cache):
        try:
            z = np.load(cache)
            if np.array_equal(z["key"], key):
                return {k: z[k] for k in ("pid", "xyz", "rgb", "err", "off", "img")}
        except Exception:
            pass
    d = _parse_points3D(path)
    try:
        tmp = f"{cache}.tmp{os.getpid()}.npz"
        np.savez(tmp, key=key, **d)
        os.replace(tmp, cache)
    except Exception as e:  # 寫不進去（唯讀）就每次重算
        print(f"[colmap-fast] ⚠ 快取寫不進 {cache}：{e}", flush=True)
    return d


def read_points3D_xyz_rgb_err(path, selected_image_ids: dict = None):
    """與 ColmapDataParser.read_points3D_binary 相同的回傳（xyz f8、rgb int64、error f8；過濾後順序不變）。"""
    d = points3D_arrays(path)
    keep = slice(None)
    if selected_image_ids is not None:
        sel = np.fromiter((int(k) for k in selected_image_ids), dtype=np.int64)
        hit = np.isin(d["img"].astype(np.int64), sel)
        cnt = np.add.reduceat(hit.astype(np.int64), d["off"][:-1]) if len(hit) else np.zeros(len(d["xyz"]), np.int64)
        cnt[np.diff(d["off"]) == 0] = 0          # reduceat 對空區段回傳下一段的值 => 長度 0 的 track 一律不選
        keep = cnt > 0
    return d["xyz"][keep], d["rgb"][keep].astype(np.int64), d["err"][keep]
