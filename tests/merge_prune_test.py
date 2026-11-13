"""訓練中合併的 merge_prune（2026-11-02：合併完就刪 j、N 減少、cap 扣掉）——純 CPU。"""
import math
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl


class _FakeGM:
    """只提供 controller／Utils 會用到的介面（properties、gaussians、get_property、n_gaussians）。"""

    def __init__(self, props):
        self._p = dict(props)

    @property
    def properties(self):
        return self._p

    @properties.setter
    def properties(self, v):
        self._p = dict(v)

    @property
    def gaussians(self):
        return self._p

    @property
    def property_names(self):
        return list(self._p.keys())

    def get_property(self, k):
        return self._p[k]

    @property
    def n_gaussians(self):
        return self._p["means"].shape[0]


def _plane(n_side=20, spacing=0.01):
    """z=0 平面上的格點面元：同色、不透明、法向朝 z、長軸＝半個格距 => 相鄰兩顆可以合併。"""
    g = torch.arange(n_side, dtype=torch.float) * spacing
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    n = xx.numel()
    props = {
        "means": torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(n)], -1),
        "scales": torch.full((n, 2), math.log(spacing / 2)),
        "rotations": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "opacities": torch.full((n, 1), math.log(0.9 / 0.1)),
        "shs_dc": torch.zeros(n, 1, 3),
        "shs_rest": torch.zeros(n, 15, 3),
    }
    props = {k: torch.nn.Parameter(v) for k, v in props.items()}
    opt = torch.optim.Adam([{"params": [v], "name": k, "lr": 1e-3} for k, v in props.items()])
    sum((v ** 2).sum() for v in props.values()).backward()
    opt.step()                                          # 讓每個參數都有 Adam 狀態（exp_avg／exp_avg_sq）
    opt.zero_grad()
    return _FakeGM(props), [opt]


def _impl(**cfg):
    c = dict(merge_k=12, merge_max_frac=0.2, merge_color_tol=1.0, merge_from_iter=3000, merge_until_iter=-1, merge_prune=True)
    c.update(cfg)
    impl = object.__new__(MCMC2DGSDensityControllerImpl)
    impl.config = SimpleNamespace(**c)
    impl._max_radii2D = torch.zeros(1)
    impl._max_tiles2D = torch.zeros(1)
    return impl


class MergePruneTest(unittest.TestCase):
    def test_prune_removes_j_and_cuts_cap(self):
        gm, opts = _plane()
        impl = _impl()
        n0 = gm.n_gaussians
        impl._absgrad_accum = torch.arange(n0, dtype=torch.float)
        out = impl._merge_event(gm, opts, 3000, prune=True)
        self.assertIsNone(out)
        cut = impl._cap_cut
        self.assertGreater(cut, 0)
        self.assertEqual(gm.n_gaussians, n0 - cut)
        for group in opts[0].param_groups:              # 參數與 Adam 狀態一起裁（只裁參數會靜默壞掉）
            p = group["params"][0]
            self.assertEqual(p.shape[0], n0 - cut, group["name"])
            st = opts[0].state[p]
            self.assertEqual(st["exp_avg"].shape[0], n0 - cut, group["name"])
            self.assertIs(gm.properties[group["name"]], p)
        self.assertIsNone(impl._max_radii2D)
        self.assertIsNone(impl._max_tiles2D)
        self.assertEqual(impl._absgrad_accum.shape[0], n0 - cut)    # |g| 跟著剪
        # 第二次累加
        impl._merge_event(gm, opts, 4500, prune=True)
        self.assertEqual(gm.n_gaussians, n0 - impl._cap_cut)
        self.assertGreater(impl._cap_cut, cut)

    def test_relocate_mode_keeps_n(self):
        gm, opts = _plane()
        impl = _impl(merge_prune=False)
        n0 = gm.n_gaussians
        out = impl._merge_event(gm, opts, 3000, prune=False)
        self.assertEqual(gm.n_gaussians, n0)
        self.assertEqual(out.shape[0], n0)
        self.assertGreater(int(out.sum()), 0)
        self.assertEqual(int(getattr(impl, "_cap_cut", 0) or 0), 0)

    def test_window(self):
        impl = _impl(merge_from_iter=16500, merge_until_iter=27000)
        self.assertFalse(impl._merge_window(15000))
        self.assertTrue(impl._merge_window(16500))
        self.assertTrue(impl._merge_window(27000))
        self.assertFalse(impl._merge_window(28500))
        self.assertTrue(_impl(merge_from_iter=3000, merge_until_iter=-1)._merge_window(29000))


class AbsorbStepTest(unittest.TestCase):
    """merge_ops.absorb_step（合併優先的 trim）與重構後的 merge_step。"""

    def test_merge_step_pair_goes_to_midpoint(self):
        from internal.density_controllers.merge_ops import merge_step
        gm, _ = _plane()
        X0 = gm.gaussians["means"].detach().clone()
        I, J, info = merge_step(gm, k=12, max_frac=0.2, color_tol=1.0)
        self.assertGreater(I.numel(), 0)
        mid = 0.5 * (X0[I] + X0[J])                                      # 同 opacity、同大小 => 權重相等
        self.assertTrue(torch.allclose(gm.gaussians["means"].detach()[I], mid, atol=1e-6))

    def test_absorb_into_coplanar_neighbors_only(self):
        from internal.density_controllers.merge_ops import absorb_step
        gm, _ = _plane()
        n = gm.n_gaussians
        g = gm.gaussians
        with torch.no_grad():
            g["means"][399] = torch.tensor([5.0, 5.0, 5.0])               # 孤立在平面外
            g["opacities"][398] = math.log(0.02 / 0.98)                     # 幾乎透明 => 不併
        src = torch.zeros(n, dtype=torch.bool)
        src[::7] = True
        src[398] = True; src[399] = True
        X0 = g["means"].detach().clone()
        I, J, info = absorb_step(gm, src, k=12, color_tol=0.1)
        self.assertGreater(J.numel(), 0)
        self.assertTrue(bool(src[J].all()))                               # 被吸收的都是 trim 候選
        self.assertFalse(bool(src[I].any()))                              # 吸收者都不是候選（會留下來）
        self.assertEqual(int(torch.unique(I).numel()), int(I.numel()))   # 每個鄰居最多吸收一顆
        self.assertNotIn(399, J.tolist())
        self.assertNotIn(398, J.tolist())
        self.assertTrue(torch.allclose(g["means"].detach()[I], 0.5 * (X0[I] + X0[J]), atol=1e-6))
        self.assertTrue(torch.equal(g["means"].detach()[J], X0[J]))      # 候選本身不動（呼叫端照常剪掉）

    def test_absorb_nothing_when_no_candidates(self):
        from internal.density_controllers.merge_ops import absorb_step
        gm, _ = _plane()
        I, J, info = absorb_step(gm, torch.zeros(gm.n_gaussians, dtype=torch.bool))
        self.assertEqual(I.numel(), 0)
        self.assertEqual(J.numel(), 0)


if __name__ == "__main__":
    unittest.main()


class MergeRenderEpsTest(unittest.TestCase):
    """★ 11-07 晚 MCMC 一致版合併：不透明質量守恆的 opacity、合併前後渲染的相對 L2 誤差、不設額度。"""

    @staticmethod
    def _two(d, o=0.6, s=0.01):
        from internal.density_controllers.merge_ops import _pair_params
        X = torch.tensor([[-d / 2, 0., 0.], [d / 2, 0., 0.]])
        S = torch.full((2, 2), s); t1 = torch.tensor([[1., 0, 0]] * 2); t2 = torch.tensor([[0., 1, 0]] * 2)
        nrm = torch.tensor([[0., 0, 1]] * 2); O = torch.full((2,), o)
        ar = torch.arange(2)
        return {m: _pair_params(O, S, t1, t2, nrm, X, ar, ar[:1], ar[1:], m) for m in ("mass", "union")}

    def test_mass_opacity_conserves_mass(self):
        p = self._two(0.02)["mass"]                                     # 並排：合併後 s1 = sqrt(s² + (d/2)²)、s2 = s
        s1, s2 = float(p["sc"][0, 0]), float(p["sc"][0, 1])
        self.assertAlmostEqual(s1, (0.01 ** 2 + 0.01 ** 2) ** 0.5, places=6)
        self.assertAlmostEqual(float(p["on"][0]) * s1 * s2, 2 * 0.6 * 0.01 * 0.01, places=9)

    def test_error_small_when_adjacent_large_when_far(self):
        near, mid, far = self._two(0.02), self._two(0.03), self._two(0.06)               # 中心距 2s／3s／6s（s＝尺度）
        self.assertLess(float(near["mass"]["err"][0]), 0.05)                               # 緊鄰：一顆幾乎畫得出兩顆
        self.assertLess(float(mid["mass"]["err"][0]), float(mid["union"]["err"][0]))       # 3s（現行幾何條件的上限）：聯集的 opacity 讓質量變多
        self.assertGreater(float(far["mass"]["err"][0]), 0.4)                              # 離太遠：一顆蓋不出兩顆的樣子

    def test_merge_step_eps_no_quota(self):
        from internal.density_controllers.merge_ops import merge_step
        gm, _ = _plane()
        I, J, info = merge_step(gm, k=12, max_frac=1.0, color_tol=1.0, opacity_mode="mass", render_eps=0.5)
        self.assertGreater(info["mutual"], 0)
        self.assertEqual(I.numel(), info["pairs"])
        self.assertLessEqual(info["pairs"], info["mutual"])
        gm2, _ = _plane()
        _, _, info0 = merge_step(gm2, k=12, max_frac=1.0, color_tol=1.0, opacity_mode="mass", render_eps=1e-6)
        self.assertEqual(info0["pairs"], 0)                                                 # 門檻極小 => 一對都不併


class MergeAsTrimTest(unittest.TestCase):
    """★ 2026-11-13 merge_as_trim（使用者「一樣用合併的門檻，分別使用合併跟 trim」）：同樣的配對、只刪不併。"""

    def test_same_pairs_no_param_change_drop_smaller(self):
        from internal.density_controllers.merge_ops import merge_step
        gm_m, _ = _plane(); gm_t, _ = _plane()
        with torch.no_grad():                                                   # 讓每對大小不同 => 看留下的是不是大的那顆
            sc = torch.linspace(-0.2, 0.2, gm_t.n_gaussians)[:, None]
            gm_t.gaussians["scales"].add_(sc); gm_m.gaussians["scales"].add_(sc)
        before = {k: v.detach().clone() for k, v in gm_t.gaussians.items()}
        Im, Jm, _ = merge_step(gm_m, k=12, max_frac=0.2, color_tol=1.0)
        It, Jt, info = merge_step(gm_t, k=12, max_frac=0.2, color_tol=1.0, as_trim=True)
        self.assertTrue(info.get("trim"))
        self.assertEqual(Im.numel(), It.numel())                               # 同一批配對
        self.assertEqual(set(map(frozenset, zip(Im.tolist(), Jm.tolist()))), set(map(frozenset, zip(It.tolist(), Jt.tolist()))))
        for k, v in gm_t.gaussians.items():                                     # 一個參數都沒改
            self.assertTrue(torch.equal(v.detach(), before[k]), k)
        S = torch.exp(before["scales"]); O = torch.sigmoid(before["opacities"]).reshape(-1)
        w = O * S[:, 0] * S[:, 1]
        self.assertTrue(bool((w[It] >= w[Jt]).all()))                           # 留下不透明質量大的

    def test_event_prunes_same_count_keeps_adam_state(self):
        gm_m, opts_m = _plane(); gm_t, opts_t = _plane()
        n0 = gm_t.n_gaussians
        impl_m = _impl(); impl_t = _impl(merge_as_trim=True)
        impl_m._merge_event(gm_m, opts_m, 3000, prune=True)
        st_before = {g["name"]: opts_t[0].state[g["params"][0]]["exp_avg"].clone() for g in opts_t[0].param_groups}
        means_before = gm_t.gaussians["means"].detach().clone()
        impl_t._merge_event(gm_t, opts_t, 3000, prune=True)
        self.assertEqual(impl_t._cap_cut, impl_m._cap_cut)                      # 同顆數、cap 一樣扣
        self.assertEqual(gm_t.n_gaussians, n0 - impl_t._cap_cut)
        # 留下的顆粒：位置與 Adam 狀態都是原本那一份（只是刪掉了一些列）
        kept = gm_t.gaussians["means"].detach()
        idx = torch.cdist(kept, means_before).argmin(1)
        self.assertTrue(torch.equal(kept, means_before[idx]))
        p = gm_t.gaussians["means"]
        self.assertTrue(torch.equal(opts_t[0].state[p]["exp_avg"], st_before["means"][idx]))


class MergeBorrowedTest(unittest.TestCase):
    """★ 2026-11-13 借鑑三點：L2 最佳不透明度、局部紋理一致才合併、線狀合併（燈桿上下段）。"""

    def test_l2_opacity_has_smallest_error(self):
        from internal.density_controllers.merge_ops import merge_step
        errs = {}
        for mode in ("union", "mass", "l2"):
            gm, _ = _plane()
            with torch.no_grad():
                gm.gaussians["opacities"].copy_(torch.logit(torch.linspace(0.55, 0.95, gm.n_gaussians))[:, None])
            _, _, info = merge_step(gm, k=12, max_frac=1.0, color_tol=1.0, opacity_mode=mode, render_eps=10.0)
            errs[mode] = info["err_med"]
        self.assertLessEqual(errs["l2"], errs["union"] + 1e-9)
        self.assertLessEqual(errs["l2"], errs["mass"] + 1e-9)

    def test_texture_gate_blocks_textured_plane(self):
        from internal.density_controllers.merge_ops import merge_step
        gm_s, _ = _plane(); gm_t, _ = _plane()
        with torch.no_grad():                                                   # 棋盤格顏色＝高紋理
            n = gm_t.n_gaussians; side = int(round(n ** 0.5))
            chk = ((torch.arange(n) // side + torch.arange(n) % side) % 2).float()
            gm_t.gaussians["shs_dc"][:, 0, :] = (chk[:, None] - 0.5) / 0.28209479177387814 * 0.6
        I_s, _, _ = merge_step(gm_s, k=12, max_frac=1.0, color_tol=1.0, tex_tol=0.05)
        I_t, _, _ = merge_step(gm_t, k=12, max_frac=1.0, color_tol=1.0, tex_tol=0.05)
        self.assertGreater(I_s.numel(), 0)                                      # 平滑的照樣併
        self.assertEqual(I_t.numel(), 0)                                        # 高紋理的不併

    def test_line_merge_pole_segments(self):
        from internal.density_controllers.merge_ops import merge_step
        def pole():
            n = 120                                                             # 一根沿 z 的燈桿：面元朝 +x、上下相接（merge_step 要 >= 100 顆候選）
            jit = torch.randn(n, 2, generator=torch.Generator().manual_seed(0)) * 0.0004    # 桿子有粗細：x、y 小抖動（完全共線時 PCA 退化）
            props = {"means": torch.stack([jit[:, 0], jit[:, 1], torch.arange(n, dtype=torch.float) * 0.01], -1),
                     "scales": torch.log(torch.tensor([[0.006, 0.002]])).repeat(n, 1),
                     # 四元數：繞 y 轉 90°（面元 t1→−z…，法向 → x）
                     "rotations": torch.tensor([[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]]).repeat(n, 1),
                     "opacities": torch.full((n, 1), math.log(0.9 / 0.1)), "shs_dc": torch.zeros(n, 1, 3), "shs_rest": torch.zeros(n, 15, 3)}
            props = {k: torch.nn.Parameter(v) for k, v in props.items()}
            return _FakeGM(props)
        I0, _, _ = merge_step(pole(), k=8, max_frac=1.0, color_tol=1.0)
        gm = pole()
        I1, J1, info = merge_step(gm, k=8, max_frac=1.0, color_tol=1.0, line=True)
        self.assertLess(I0.numel(), 6)                                          # 平面規則對線狀鄰域幾乎都不過（偶爾抖動湊出一個扁鄰域）
        self.assertGreater(I1.numel(), 5 * max(I0.numel(), 1))                  # 線狀規則把上下段併起來
        self.assertGreater(info["line_pairs"], 0)
        from internal.density_controllers.merge_ops import _qmat
        R = _qmat(gm.gaussians["rotations"].detach()[I1])
        S = torch.exp(gm.gaussians["scales"].detach()[I1])
        long_axis = torch.where((S[:, :1] >= S[:, 1:]), R[:, :, 0], R[:, :, 1])
        self.assertTrue(bool((long_axis[:, 2].abs() > 0.95).all()))             # 併完的長軸沿桿子（z）
        self.assertTrue(bool((S.max(1).values > 0.006).all()))                  # 變長了
