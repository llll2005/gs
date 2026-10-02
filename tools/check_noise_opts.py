"""fast_noise／noise_gate 在新資料上是否無害（2026-10-02，取代 2 x 60k 的訓練驗證）。

兩者都只改 MCMC 的位置噪音（mcmc_2dgs_density_controller._add_xyz_noise）：
  fast_noise   cov @ v 改寫成 R (s^2 * (R^T v))：數學等價，只差浮點 => 用同一組隨機 v 比兩種算法
  noise_gate   gate = op_sigmoid(1-o) <= eps 的顆粒不加噪音：量「被跳過者本來每步會移多少、相對自身尺寸」
               以及 60k 步隨機遊走累積（sqrt(T) 倍）的上界
用訓練當下的 means 學習率（ckpt 的 optimizer param_groups）與 config 的 noise_lr，與訓練用同一個 op_sigmoid。
用法：python tools/check_noise_opts.py <ckpt> [<ckpt> ...] [--eps 1e-3]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def op_sigmoid(x, k=100, x0=0.995):          # 與 mcmc_density_controller.op_sigmoid 相同
    return 1 / (1 + torch.exp(-k * (x - x0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--steps-left", type=int, default=60000)
    a = ap.parse_args()
    from internal.utils.gaussian_projection import compute_cov_3d, build_rotation_matrix
    torch.manual_seed(0)
    for p in a.ckpts:
        ck = torch.load(p, map_location="cpu")
        sd = ck["state_dict"]
        pre = "gaussian_model.gaussians."
        o = torch.sigmoid(sd[pre + "opacities"].float()).reshape(-1)
        s2 = torch.exp(sd[pre + "scales"].float())
        q = torch.nn.functional.normalize(sd[pre + "rotations"].float(), dim=-1)
        N = o.numel()
        scales = torch.cat([s2, torch.zeros(N, 3 - s2.shape[1])], -1) if s2.shape[1] < 3 else s2
        lr, adam = None, None
        for opt in ck["optimizer_states"]:
            for grp in opt["param_groups"]:
                if grp.get("name") == "means":
                    lr = grp["lr"]
                    st = opt["state"].get(grp["params"][0])
                    if st is not None:                    # 同一顆的 Adam 步長 |lr * m_hat / (sqrt(v_hat)+eps)|
                        b1, b2 = grp.get("betas", (0.9, 0.999))
                        t = float(st["step"])
                        m = st["exp_avg"].float() / (1 - b1 ** t)
                        vv = st["exp_avg_sq"].float() / (1 - b2 ** t)
                        adam = (lr * m / (vv.sqrt() + grp.get("eps", 1e-15))).norm(dim=-1)
        dens = ck["hyper_parameters"]["density"]
        noise_lr = float(getattr(dens, "noise_lr", 5e5))
        g = op_sigmoid(1 - o)
        v = torch.randn(N, 3) * (g * noise_lr * lr).unsqueeze(-1)
        # fast_noise vs 原算法
        cov = compute_cov_3d(scales=scales, scale_modifier=1., quaternions=q)
        a_ = torch.bmm(cov, v.unsqueeze(-1)).squeeze(-1)
        R = build_rotation_matrix(q)
        b_ = torch.bmm(R, (torch.bmm(R.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1) * scales ** 2).unsqueeze(-1)).squeeze(-1)
        d = (a_ - b_).abs()
        na = a_.norm(dim=-1)
        rel = (d.norm(dim=-1) / na.clamp_min(1e-30))
        big = na > torch.quantile(na[na > 0], 0.5) if bool((na > 0).any()) else torch.zeros_like(na, dtype=torch.bool)
        # noise_gate：被跳過者
        skip = g <= a.eps
        size = scales.max(dim=-1).values
        step_disp = na / size.clamp_min(1e-30)                   # 每步位移 / 自身尺寸
        walk = step_disp * (a.steps_left ** 0.5)
        print(f"== {os.path.basename(os.path.dirname(os.path.dirname(p)))}/{os.path.basename(p)}  N={N:,}  means_lr={lr:.3e}  noise_lr={noise_lr:g}")
        print(f"   fast_noise：最大絕對差 {float(d.max()):.3e}（噪音本身最大 {float(na.max()):.3e}）"
              f"  相對差 中位 {float(rel[na > 0].median()):.2e}、對噪音較大的那一半的 p99 {float(torch.quantile(rel[big], 0.99)) if bool(big.any()) else float('nan'):.2e}"
              f"  兩法的位移範數平均 {float(na.mean()):.3e} vs {float(b_.norm(dim=-1).mean()):.3e}")
        ws = walk[skip]
        if adam is not None and adam.numel() == N and bool(skip.any()):
            r = na[skip] / adam[skip].clamp_min(1e-30)
            print(f"   noise_gate：被跳過者「噪音位移 / 同顆 Adam 步長」 中位 {float(r.median()):.2e}  p99 {float(torch.quantile(r, 0.99)):.2e}  最大 {float(r.max()):.2e}"
                  f"（> 0.1 的顆數 {int((r > 0.1).sum()):,}）")
        print(f"   noise_gate eps={a.eps:g}：跳過 {100 * float(skip.float().mean()):.1f}%；被跳過者每步位移/自身尺寸 最大 {float(step_disp[skip].max()) if bool(skip.any()) else 0:.2e}，"
              f"{a.steps_left} 步隨機遊走累積 最大 {float(ws.max()) if ws.numel() else 0:.2e}、p99 {float(torch.quantile(ws, 0.99)) if ws.numel() else 0:.2e}"
              f"（對照：沒被跳過者每步中位 {float(step_disp[~skip].median()) if bool((~skip).any()) else 0:.2e}）")
        del ck, sd, cov, R, a_, b_, adam


if __name__ == "__main__":
    main()
