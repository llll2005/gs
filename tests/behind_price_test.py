"""★ 2026-11-11 規格 2 §3 落後貢獻計價（renderer trim_behind_price、density sample_behind_price）——純 CPU。"""
import unittest
from types import SimpleNamespace

import torch

from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer, behind_rho, behind_price_factor


class RhoTest(unittest.TestCase):
    def test_rho_and_evidence(self):
        Vc = torch.tensor([10.0, 10.0, 0.5, 0.0, 10.0])
        Vb = torch.tensor([5.0, 0.0, 0.5, 0.0, 10.0])
        rho, ok = behind_rho(Vc, Vb, 0.1)                       # 中位（Vc>0）＝10 => 門檻 1 => 第 3 顆證據不足
        self.assertEqual(ok.tolist(), [True, True, False, False, True])
        self.assertTrue(torch.allclose(rho, torch.tensor([0.5, 0.0, 0.0, 0.0, 1.0])))

    def test_factor_floor_and_off(self):
        Vc = torch.tensor([10.0, 10.0, 10.0]); Vb = torch.tensor([10.0, 5.0, 0.0])
        self.assertIsNone(behind_price_factor(Vc, Vb, 0.0))
        f, _, _ = behind_price_factor(Vc, Vb, 1.0, 0.1, 0.01)
        self.assertTrue(torch.allclose(f, torch.tensor([0.01, 0.5, 1.0])))
        f, _, _ = behind_price_factor(Vc, Vb, 1.0, 0.1, 0.0)
        self.assertTrue(torch.allclose(f, torch.tensor([0.0, 0.5, 1.0])))


class _PC:
    def __init__(self, V, C, BVc=None, BVb=None):
        self.get_xyz = torch.zeros(V.shape[0], 3)
        self._trim_state = {"V": V.clone(), "C": C.clone(), "views": 3}
        if BVc is not None:
            self._trim_state["BVc"] = BVc.clone(); self._trim_state["BVb"] = BVb.clone()

    def get_opacities(self):
        return torch.ones(self.get_xyz.shape[0], 1)


def _trim(V, C, lam=0.0, BVc=None, BVb=None, ratio=0.25):
    r = object.__new__(SepDepthTrim2DGSRenderer)
    r.prune_ratio = ratio; r.trim_running_score = "vc"; r.trim_protect_vc = -1.0; r.trim_soft_rho = 0.0; r.trim_merge_first = False
    r.trim_behind_price = lam; r.behind_price_evidence = 0.1; r.behind_price_decay = 0.5
    got = {}
    pc = _PC(V, C, BVc, BVb)
    dc = SimpleNamespace(_prune_points=lambda mask, pc_, opts: got.setdefault("mask", mask.clone()))
    r._running_trim(5000, SimpleNamespace(gaussian_model=pc, density_controller=dc, gaussian_optimizers=[]), 15000)
    return got["mask"], pc


class TrimPriceTest(unittest.TestCase):
    V = torch.tensor([10., 20., 30., 40., 50., 25., 16., 0.])     # V/C＝[1, 2, 3, 4, 0.5, 0.25, 8, —]
    C = torch.tensor([10., 10., 10., 10., 100., 100., 2., 0.])

    def test_lambda_zero_identical(self):
        m0, _ = _trim(self.V, self.C)
        BVc = torch.full((8,), 10.0); BVb = torch.tensor([0., 0, 0, 10, 0, 0, 10, 0])
        m1, pc = _trim(self.V, self.C, 0.0, BVc, BVb)
        self.assertTrue(torch.equal(m0, m1))
        self.assertTrue(torch.equal(pc._trim_state["BVc"], BVc * 0.5))   # 帳照樣衰減（λ＝0 也不歸零）

    def test_priced_prunes_behind_first(self):
        BVc = torch.full((8,), 10.0); BVb = torch.tensor([0., 0, 0, 10, 0, 0, 10, 0])   # 第 4、7 顆全在表面後面（V/C 4、8）
        m, pc = _trim(self.V, self.C, 1.0, BVc, BVb)
        self.assertEqual(m.nonzero().flatten().tolist(), [3, 6])          # λ＝1：落後的貢獻不算價值 => 分數 0 先剪
        self.assertFalse(bool(m[7]))                                       # 沒被看到的照舊不剪
        self.assertTrue(torch.allclose(pc._trim_state["BVb"], BVb * 0.5))


class SamplePriceTest(unittest.TestCase):
    def _impl(self, lam):
        from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl
        impl = object.__new__(MCMC2DGSDensityControllerImpl)
        impl.config = SimpleNamespace(sample_behind_price=lam, behind_price_evidence=0.1)
        return impl

    def _gm(self, n, BVc=None, BVb=None):
        st = {} if BVc is None else {"BVc": BVc, "BVb": BVb}
        return SimpleNamespace(n_gaussians=n, _trim_state=st)

    def test_off_returns_same(self):
        p = torch.rand(5)
        self.assertIs(self._impl(0.0)._behind_price_probs(self._gm(5), p), p)

    def test_factor_and_tail_padding(self):
        BVc = torch.tensor([10.0, 10.0, 10.0]); BVb = torch.tensor([10.0, 5.0, 0.0])
        out = self._impl(1.0)._behind_price_probs(self._gm(5, BVc, BVb), torch.ones(5))   # 後 2 顆是新增的、沒帳
        self.assertTrue(torch.allclose(out, torch.tensor([0.01, 0.5, 1.0, 1.0, 1.0])))
        idx = torch.tensor([0, 2])                                         # 搬遷路徑（只抽活的）
        out2 = self._impl(1.0)._behind_price_probs(self._gm(5, BVc, BVb), torch.ones(2), idx)
        self.assertTrue(torch.allclose(out2, torch.tensor([0.01, 1.0])))

    def test_no_accounts_is_noop(self):
        p = torch.rand(4)
        self.assertTrue(torch.equal(self._impl(1.0)._behind_price_probs(self._gm(4), p), p))


class AccountSyncTest(unittest.TestCase):
    def test_trim_state_pads_and_prunes_accounts(self):
        pc = SimpleNamespace(get_xyz=torch.zeros(4, 3))
        st = SepDepthTrim2DGSRenderer._trim_state(pc, ("BVc", "BVb"))
        st["BVc"] += 1
        pc.get_xyz = torch.zeros(6, 3)                                     # 增生：尾端補 0
        st = SepDepthTrim2DGSRenderer._trim_state(pc, ("BVc", "BVb"))
        self.assertEqual(st["BVc"].tolist(), [1, 1, 1, 1, 0, 0])
        from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl
        impl = object.__new__(MCMC2DGSDensityControllerImpl)
        impl._anchor_mask = lambda n: None; impl._anchor_fate = lambda m, r: None; impl._PER_PARTICLE = ()
        import internal.density_controllers.mcmc_2dgs_density_controller as mod
        orig = mod.Utils.prune_properties
        mod.Utils.prune_properties = staticmethod(lambda valid, gm, opts: {})
        try:
            gm = SimpleNamespace(_trim_state=st, properties={})
            impl._prune_points(torch.tensor([True, False, False, True, False, False]), gm, [])
        finally:
            mod.Utils.prune_properties = orig
        self.assertEqual(gm._trim_state["BVc"].tolist(), [1, 1, 0, 0])   # 剪枝跟著同步


if __name__ == "__main__":
    unittest.main()
