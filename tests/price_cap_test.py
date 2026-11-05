"""定價自決容量（density price_*；2026-11-05 使用者「做一個通用、更省的自適應 cap」）——純 CPU。"""
import unittest
from types import SimpleNamespace

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl


def _impl(**cfg):
    c = dict(price_lambda=0.01, price_probation=500, price_headroom=0.9, price_cap_min=10, price_from_iter=3000,
             price_stop_tol=0.0, price_stop_from=6000, price_stop_patience=3, cap_max=1000, densify_until_iter=15000)
    c.update(cfg)
    impl = object.__new__(MCMC2DGSDensityControllerImpl)
    impl.config = SimpleNamespace(**c)
    return impl


def _vc(n, n_good, n_unseen=0):
    """前 n_good 顆 V/C = 0.02（> λ 0.01）、其餘 0.005；最後 n_unseen 顆這段沒被看到（C = 0）。"""
    C = torch.full((n,), 10.0)
    V = torch.full((n,), 0.05)
    V[:n_good] = 0.2
    if n_unseen:
        C[-n_unseen:] = 0.0
        V[-n_unseen:] = 0.0
    return V, C, C > 0


class PriceCapTest(unittest.TestCase):
    def test_cap_follows_value(self):
        impl = _impl()
        V, C, seen = _vc(500, 300)
        impl.price_update(3000, None, V, C, seen, torch.zeros(500, dtype=torch.bool))
        self.assertEqual(impl._price_cap, int(300 / 0.9))
        self.assertEqual(impl.config.cap_max - impl._cap_cut, impl._price_cap)   # add_new_gs 的上限＝cap_t

    def test_unknown_counts_as_ok_and_pruned_excluded(self):
        impl = _impl()
        V, C, seen = _vc(500, 100, n_unseen=50)
        low = torch.zeros(500, dtype=torch.bool)
        low[:20] = True                                   # 這輪要剪的（即使 V/C 高也不算）
        b = torch.full((500,), -1, dtype=torch.long)
        b[100:130] = 2800                                 # 剛出生（試用期內）=> 未知
        impl._birth = b
        impl.price_update(3000, None, V, C, seen, low)
        n_ok = (100 - 20) + 30 + 50                       # 合格 80（扣掉被剪的 20）＋ 試用期 30 ＋ 沒看到 50
        self.assertEqual(impl._price_cap, int(n_ok / 0.9))

    def test_clipped_to_hardware_and_floor(self):
        impl = _impl(cap_max=200)
        V, C, seen = _vc(500, 500)
        impl.price_update(3000, None, V, C, seen, torch.zeros(500, dtype=torch.bool))
        self.assertEqual(impl._price_cap, 200)
        impl = _impl(price_cap_min=150)
        V, C, seen = _vc(500, 10)
        impl.price_update(3000, None, V, C, seen, torch.zeros(500, dtype=torch.bool))
        self.assertEqual(impl._price_cap, 150)

    def test_inactive_before_from_iter_and_when_off(self):
        impl = _impl()
        V, C, seen = _vc(500, 300)
        impl.price_update(2500, None, V, C, seen, torch.zeros(500, dtype=torch.bool))
        self.assertFalse(hasattr(impl, "_price_cap"))
        impl = _impl(price_lambda=0.0)
        impl.price_update(5000, None, V, C, seen, torch.zeros(500, dtype=torch.bool))
        self.assertFalse(hasattr(impl, "_price_cap"))

    def test_early_stop_when_converged(self):
        impl = _impl(price_stop_tol=0.02)
        z = torch.zeros(500, dtype=torch.bool)
        for i, step in enumerate(range(3000, 9001, 500)):
            V, C, seen = _vc(500, 300 if step < 5000 else 400)   # 5000 起容量跳一次，之後穩定
            impl.price_update(step, None, V, C, seen, z)
            if impl.config.densify_until_iter != 15000:
                break
        # 6000 前不停（price_stop_from）；穩定 3 次（5500、6000、6500 的變動都 0）=> 6500 停
        self.assertEqual(impl.config.densify_until_iter, 6500)

    def test_no_early_stop_without_tol(self):
        impl = _impl()
        V, C, seen = _vc(500, 300)
        for step in range(3000, 12001, 500):
            impl.price_update(step, None, V, C, seen, torch.zeros(500, dtype=torch.bool))
        self.assertEqual(impl.config.densify_until_iter, 15000)

    def test_cap_init_before_price(self):
        impl = _impl(price_cap_init=600, cap_max=1000)
        called = {}
        def fake_super_add(gm, opts):                     # 只檢查交給 MCMC 的上限
            called["cap"] = impl.config.cap_max - int(getattr(impl, "_cap_cut", 0) or 0)
        import internal.density_controllers.mcmc_density_controller as base
        orig = base.MCMCDensityControllerImpl.add_new_gs
        base.MCMCDensityControllerImpl.add_new_gs = lambda self, gm, opts: fake_super_add(gm, opts)
        try:
            gm = SimpleNamespace(n_gaussians=100)
            impl._inplane_on = lambda: False
            impl.add_new_gs(gm, None)
        finally:
            base.MCMCDensityControllerImpl.add_new_gs = orig
        self.assertEqual(called["cap"], 600)


if __name__ == "__main__":
    unittest.main()
