"""比較各臂的梯度：逐參數相對 L2、與基準差超過門檻的顆粒數（只 1 個視角、增生期設定）。"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from internal.utils.gaussian_model_loader import GaussianModelLoader
ck_path = sys.argv[1]; vi = int(sys.argv[2]) if len(sys.argv) > 2 else 300
dev = torch.device("cuda")
model, renderer, ck = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck_path, device=dev, eval_mode=False, pre_activate=False)
dmh = ck["datamodule_hyper_parameters"]; _pp = dmh["parser"]
ts = dmh["parser"].instantiate(path=dmh["path"], output_path=os.path.dirname(os.path.dirname(ck_path)), global_rank=0).get_outputs().train_set
cam = ts.cameras[vi].to_device(dev); del ck
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
bg = torch.zeros(3, device=dev); renderer._acc_stop = False
torch.manual_seed(0); Wt = None
def grads(f, pb):
    global Wt
    os.environ["CITYGS_SKIPF"] = str(f); renderer.bwd_pair_bit = pb
    for _, p in named: p.grad = None
    out = renderer(cam, model, bg_color=bg, _lean=True, _absgrad=True)
    img = out["render"]
    if Wt is None: Wt = torch.rand(img.shape, device=dev) - 0.5
    (img * Wt).sum().backward(); torch.cuda.synchronize()
    return {n: p.grad.detach().cpu().clone() for n, p in named}
base = grads(0, False)
for name, f, pb in [("雜訊（基準重跑）", 0, False), ("A-1 bwd", 2, False), ("配對 bit（現有）", 0, True), ("A-3", 4, True)]:
    g = grads(f, pb)
    rows = []
    for n, g0 in base.items():
        d = (g[n] - g0).reshape(g0.shape[0], -1); g0r = g0.reshape(g0.shape[0], -1)
        rel = float(d.norm() / max(g0r.norm(), 1e-30))
        bad = int(((d.abs() > 1e-4 * g0r.abs().amax(1, keepdim=True).clamp_min(1e-20)).any(1)).sum())
        rows.append(f"{n.split('.')[-1]} 相對L2 {rel:.1e}／差>1e-4 的顆 {bad:,}")
    print(f"{name:14}｜" + "｜".join(rows), flush=True)
