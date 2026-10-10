"""塊外池：單塊訓練時，把落在本塊分區盒（＋margin）之外的顆粒搬出 MCMC 的可訓練集合，用便宜的方式保留（2026-10-10 使用者 A／B 提案）。

為什麼：4x4 全場景 16 塊共訓練 37.4M 顆、合併只留 17.6M（47%）——約一半的 VRAM／時間花在合併時會丟掉的顆粒上
（`紀錄/實驗分析/09` §0.2b、記憶 out_of_block_eval_bias）。它們仍要留著：訓練視角大半畫面在塊外
（相機分配 content_threshold 0.08），沒人解釋那些像素時，梯度會推壞塊內顆粒。=> 不刪，改成便宜的計價：
  - 只留 SH0（shs_rest 丟掉）：每顆參數 232 -> 52 B
  - A（trainable=True）：池內顆粒照樣做梯度下降（自己的 Adam），但不受 MCMC 管（不佔 cap、不搬移、不加噪音、不 trim）；
                         可設尺寸下限（建立時池內最大軸尺寸的分位數），讓背景用較少、較大的顆粒
                         => 常駐 52 x 4（參數＋梯度＋Adam 兩個動量）= 208 B／顆（原本 928）
  - B（凍結）：不再更新；建立／凍結時直接存激活後的值 => 52 B／顆，沒有梯度與 Adam
MCMC 只看得到可訓練集合 => cap 只算塊內（＋margin），搬出去的名額在 densify 期會被塊內補回。
渲染：renderer 在光柵化前把可訓練集合與池串接，逐顆輸出（radii／tiles／record）切回前 N_in 筆
=> density controller、trim、absgrad 完全不知道池的存在（不會像上游 HasExtraParameters 那樣讓 MCMC 抽到池內顆粒再被截斷）。
⚠ 合併（utils/merge_citygs_ckpts.py）只讀 state_dict 裡的可訓練集合：池內顆粒都在盒＋margin 之外 => 合併時本來就會被丟掉，結果相同。

激活函數寫死成 sigmoid／exp／normalize（與 VanillaGaussianModel 相同）：池在工具裡載入時，模型可能已 pre_activate（激活被換成 identity）。
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F

KEYS = ("means", "opacities", "scales", "rotations", "shs_dc")


class OutsidePool:
    def __init__(self, raw: Dict[str, torch.Tensor], trainable: bool, min_scale: Optional[float] = None):
        self.trainable = bool(trainable)
        self.min_scale = float(min_scale) if min_scale else None
        self.raw = {k: raw[k].detach().clone().contiguous() for k in KEYS}
        self.act = None          # 凍結後：激活過的常數
        self.optimizer = None
        if self.trainable:
            self._make_params()

    # ── 基本 ──
    @property
    def n(self) -> int:
        src = self.act if self.act is not None else self.raw
        return int(src["means"].shape[0])

    @property
    def device(self):
        src = self.act if self.act is not None else self.raw
        return src["means"].device

    def to(self, device):
        if self.device == torch.device(device):      # 呼叫端傳的是渲染張量的 device（例 cuda:0），可直接比
            return self
        if self.act is not None:
            self.act = {k: v.to(device) for k, v in self.act.items()}
        else:
            was = self.trainable
            self.raw = {k: v.detach().to(device) for k, v in self.raw.items()}
            if was:
                self._make_params()
        return self

    def bytes_per_point(self) -> int:
        f = 3 + 1 + int(self.raw["scales"].shape[-1] if self.act is None else self.act["scales"].shape[-1]) + 4 + 3
        return f * 4 * (4 if self.trainable else 1)

    # ── 訓練 ──
    def _make_params(self, states=None):
        self.raw = {k: torch.nn.Parameter(v.detach().clone(), requires_grad=True) for k, v in self.raw.items()}
        lr = {"means": 1.6e-4, "opacities": 0.05, "scales": 0.005, "rotations": 0.001, "shs_dc": 0.0025}
        self.optimizer = torch.optim.Adam([{"params": [self.raw[k]], "lr": lr[k], "name": k} for k in KEYS], eps=1e-15)
        if states:
            for g in self.optimizer.param_groups:
                st = states.get(g["name"])
                if st is not None:
                    self.optimizer.state[g["params"][0]] = st

    def add(self, raw: Dict[str, torch.Tensor]):
        """把新搬出來的顆粒接到池尾（A 的週期性清掃）。Adam 動量接 0。"""
        if raw["means"].shape[0] == 0:
            return
        if self.act is not None:          # 已凍結：直接存激活值
            a = self._activate({k: raw[k].detach() for k in KEYS})
            self.act = {k: torch.cat([self.act[k], a[k]], 0) for k in KEYS}
            return
        states = {}
        if self.trainable and self.optimizer is not None:
            for g in self.optimizer.param_groups:
                p = g["params"][0]; st = self.optimizer.state.get(p, {})
                if "exp_avg" in st:
                    z = torch.zeros_like(raw[g["name"]])
                    states[g["name"]] = {"step": st["step"], "exp_avg": torch.cat([st["exp_avg"], z], 0),
                                         "exp_avg_sq": torch.cat([st["exp_avg_sq"], z], 0)}
        self.raw = {k: torch.cat([self.raw[k].detach(), raw[k].detach().to(self.raw[k].device)], 0) for k in KEYS}
        if self.trainable:
            self._make_params(states)

    def sync_lr(self, main_optimizers):
        """池的學習率跟著主模型同名參數群走（means 有排程）。"""
        if self.optimizer is None:
            return
        cur = {}
        for opt in main_optimizers:
            for g in opt.param_groups:
                if g.get("name") in KEYS:
                    cur[g["name"]] = g["lr"]
        for g in self.optimizer.param_groups:
            if g["name"] in cur:
                g["lr"] = cur[g["name"]]

    def step(self, main_optimizers):
        if self.optimizer is None:
            return
        self.sync_lr(main_optimizers)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def freeze(self):
        """B：存激活值、丟掉參數與 Adam（每顆 208 -> 52 B）。"""
        if self.act is not None:
            return
        with torch.no_grad():
            self.act = self._activate({k: v.detach() for k, v in self.raw.items()})
        self.raw = None
        self.optimizer = None
        self.trainable = False

    # ── 渲染 ──
    def _activate(self, r):
        sc = torch.exp(r["scales"])
        if self.min_scale:
            sc = torch.clamp_min(sc, self.min_scale)
        return {"means": r["means"], "opacities": torch.sigmoid(r["opacities"]), "scales": sc,
                "rotations": F.normalize(r["rotations"]), "shs_dc": r["shs_dc"]}

    def render_tensors(self, n_sh_coeffs: int):
        a = self.act if self.act is not None else self._activate(self.raw)
        dc = a["shs_dc"]
        feats = dc if n_sh_coeffs <= 1 else torch.cat(
            [dc, torch.zeros((dc.shape[0], n_sh_coeffs - 1, dc.shape[2]), dtype=dc.dtype, device=dc.device)], 1)
        return a["means"], a["opacities"], a["scales"], a["rotations"], feats

    # ── 存檔 ──
    def state(self) -> dict:
        if self.act is not None:
            return {"frozen": True, "min_scale": self.min_scale, "act": {k: v.detach().cpu() for k, v in self.act.items()}}
        return {"frozen": False, "trainable": self.trainable, "min_scale": self.min_scale,
                "raw": {k: v.detach().cpu() for k, v in self.raw.items()}}

    @classmethod
    def from_state(cls, s: dict, device=None) -> "OutsidePool":
        if s.get("frozen"):
            p = cls.__new__(cls)
            p.trainable, p.min_scale, p.raw, p.optimizer = False, s.get("min_scale"), None, None
            p.act = {k: v.to(device) if device is not None else v for k, v in s["act"].items()}
            return p
        raw = {k: v.to(device) if device is not None else v for k, v in s["raw"].items()}
        return cls(raw, trainable=s.get("trainable", False), min_scale=s.get("min_scale"))


def block_outside_mask(means: torch.Tensor, parser_cfg, margin: float) -> torch.Tensor:
    """True = 在本塊分區盒放大 margin（盒邊長的比例）之外。規則與 utils/merge_citygs_ckpts.py 相同（旋轉 -> 收縮 -> xy 盒）。"""
    import os
    from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
    parts = torch.load(os.path.join(os.path.dirname(parser_cfg.image_list), "partitions.pt"))
    coords = PartitionCoordinates(id=parts["partition_coordinates"]["id"], xy=parts["partition_coordinates"]["xy"])
    boxes = coords.get_bounding_boxes(parts["scene_config"]["partition_size"], enlarge=margin)
    x = means.detach().float().cpu() @ parts["extra_data"]["rotation_transform"][:3, :3].T.float()
    if parts["scene_config"]["contract"]:
        x = CityGSPartitioning.contract_to_unisphere(x[:, :2], parts["scene_config"]["aabb"], ord=torch.inf)
    inside = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=x[:, :2])[parser_cfg.block_id]
    return (~inside).to(means.device)
