#!/usr/bin/env python
"""★ 11-21 fused_reg 整合檢查（不訓練）：用真實 ckpt 的 metric 設定＋模型，跑 get_train_metrics 的「逐顆正則」那段，
fused_reg 關／開比 loss 與 opacity／scale 的梯度。光度 loss 與無關的子步驟換成空操作（它們不受旗標影響）。
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.metrics import citygsv2_metrics as base
from internal.metrics import mcmc_citygsv2_metrics as mm

CK = sys.argv[1] if len(sys.argv) > 1 else "outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt"
dev = torch.device("cuda")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(CK, device=dev, eval_mode=False, pre_activate=False)
ck = torch.load(CK, map_location="cpu", weights_only=False)
mcfg = ck["hyper_parameters"]["metric"]
mcfg.block_mask_dir = ""                       # 遮罩合成另外比（逐位元，見 fused_reg_bench_1121.py）
mod = mcfg.instantiate()
base.CityGSV2MetricsModule.get_train_metrics = lambda self, *a, **k: ({"loss": torch.zeros((), device=dev)}, {})
for nm in ("_see_through_step", "_depth_cap_step", "_normal_prior_step", "_normal_prior_narrow_step", "_sea_front_step", "_surf_step"):
    setattr(mod, nm, lambda *a, **k: None)
N = model.n_gaussians
g = torch.Generator(device=dev).manual_seed(0)
vf = torch.rand(N, device=dev, generator=g) < 0.6
outputs = {"visibility_filter": vf, "render": torch.zeros(3, 4, 4, device=dev)}


class PL:
    density_controller = None
    renderer = None


params = model.gaussians
for p in params.values():
    p.requires_grad_(True)


def run(step, fused):
    mod.config.fused_reg = fused
    for p in params.values():
        p.grad = None
    met, _ = mod.get_train_metrics(PL(), model, step, (None, ("x", torch.zeros(3, 4, 4, device=dev))), outputs)
    met["loss"].backward()
    return float(met["loss"]), params["opacities"].grad.clone(), params["scales"].grad.clone(), met


for step in (3000, 6000):
    l0, go0, gs0, m0 = run(step, False)
    l1, go1, gs1, m1 = run(step, True)
    x = params["opacities"].detach().reshape(-1)
    ins = (x.abs() < 13.8)[:, None]
    ro = float(((go0 - go1).abs() * ins).max() / go0.abs().max())
    rs = float((gs0 - gs1).abs().max() / gs0.abs().max())
    print(f"step {step}：loss {l0:.9f} vs {l1:.9f}（相對差 {abs(l0 - l1) / abs(l0):.1e}）｜opacity 梯度 max 相對差（|x|<13.8）{ro:.1e}｜scale 梯度 {rs:.1e}"
          f"｜op_ent {float(m0.get('op_ent', float('nan'))):.6f} vs {float(m1.get('op_ent', float('nan'))):.6f}｜sc_reg {float(m0['sc_reg']):.6e} vs {float(m1['sc_reg']):.6e}")
