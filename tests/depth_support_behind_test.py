"""★ 2026-11-11 深度支撐閘門的 behind 票（`geom_gate.DepthSupport`、`MCMCCityGSV2MetricsModule._geom_gate_step`、
`MCMCDensityControllerImpl._geom_gate_probs`）——規格 `紀錄/new_archived/規格與交接/規格_組合臂_表面起點增生限制深度上限_1111.md` §2 的四項測試＋抽樣接線。純 CPU。"""
import unittest
from types import SimpleNamespace

import torch

from internal.metrics.geom_gate import DepthSupport

# 相機在原點看 +z、fx＝fy＝20、cx＝cy＝20、40x40；表面反深度 1/2（平面 z＝2）。
CAM = (torch.eye(4), 20.0, 20.0, 20.0, 20.0, 40, 40)
# 0 面上、1 後 2%（地下層）、2 後 20%（隔著樓）、3 前方（路燈／車）、4 後 2% 但投到深度 0 的角落（不可靠像素）
MEANS = torch.tensor([[0.0, 0, 2.0], [0.0, 0, 2.04], [0.0, 0, 2.4], [0.0, 0, 1.5], [-1.2, -1.2, 2.04]])


def _inv():
    inv = torch.full((40, 40), 0.5)
    inv[:10, :10] = 0.0                     # 可靠度過濾把這塊設成 0
    return inv


def _run(ds, means, steps=7, t0=0):
    for s in range(t0, t0 + steps):
        ds.update(s, means, *CAM, _inv())
    return ds


class DepthSupportBehindTest(unittest.TestCase):
    def test_a_off_by_default_support_identical(self):
        off = _run(DepthSupport(window=5), MEANS)
        on = _run(DepthSupport(window=5, behind_margin=0.01, behind_max=0.05), MEANS)
        self.assertIsNone(off.bm)
        self.assertIsNone(off.behind)
        self.assertNotIn("behind", off.last)
        self.assertTrue(torch.equal(off.support, on.support))         # behind 不併進 support
        self.assertEqual({k: v for k, v in on.last.items() if k != "behind"}, off.last)

    def test_b_narrow_band(self):
        ds = _run(DepthSupport(window=5, behind_margin=0.01, behind_max=0.05), MEANS)
        self.assertEqual(ds.behind.tolist(), [False, True, False, False, False])
        self.assertAlmostEqual(ds.last["behind"], 1 / 5, places=6)
        # margin 收到 3% => 後 2% 的不再算
        ds3 = _run(DepthSupport(window=5, behind_margin=0.03, behind_max=0.05), MEANS)
        self.assertFalse(bool(ds3.behind[1]))
        # 只有少數視角在後面 => 不過半、不判
        ds = DepthSupport(window=5, behind_margin=0.01, behind_max=0.05)
        far = MEANS.clone(); far[1, 2] = 2.4                          # 後 20%
        for s in range(7):
            ds.update(s, MEANS if s < 2 else far, *CAM, _inv())
        self.assertFalse(bool(ds.behind[1]))

    def test_c_zero_depth_not_observed(self):
        ds = _run(DepthSupport(window=5, behind_margin=0.01, behind_max=0.05), MEANS)
        self.assertFalse(bool(ds.behind[4]))
        self.assertTrue(bool(ds.support[4]))                          # 沒有有效觀測 => 也不會被判無支撐
        # 中間步把整張深度設 0：那幾步不算觀測，cnt 只算有效的
        ds = DepthSupport(window=5, behind_margin=0.01, behind_max=0.05)
        for s in range(7):
            inv = _inv() if s % 2 == 0 else torch.zeros(40, 40)
            ds.update(s, MEANS, *CAM, inv)
        self.assertTrue(bool(ds.behind[1]))                            # 3 次有效、3 票 => 判

    def test_d_count_changes(self):
        ds = _run(DepthSupport(window=5, behind_margin=0.01, behind_max=0.05), MEANS)
        grown = torch.cat([MEANS, MEANS[1:2]])                        # 增生補在尾端
        ds.update(7, grown, *CAM, _inv())
        self.assertEqual(ds.behind.shape[0], 6)
        self.assertEqual(ds.bvote.shape[0], 6)
        self.assertTrue(bool(ds.behind[1]))                            # 舊顆粒保留判定
        self.assertFalse(bool(ds.behind[5]))                           # 新顆粒先不算
        _run(ds, grown, steps=6, t0=8)                                 # 下一個視窗 => 新顆粒也判
        self.assertTrue(bool(ds.behind[5]))
        ds.update(20, MEANS[:3], *CAM, _inv())                         # 減少 => 整個重來
        self.assertEqual(ds.behind.shape[0], 3)
        self.assertFalse(bool(ds.behind.any()))

    def test_bad_band_rejected(self):
        with self.assertRaises(ValueError):
            DepthSupport(behind_margin=0.05, behind_max=0.05)


class _Model:
    def __init__(self, xyz):
        self._xyz = xyz
        self.opacities = torch.logit(torch.full((xyz.shape[0], 1), 0.9)).clone().requires_grad_(True)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def n_gaussians(self):
        return self._xyz.shape[0]


def _cam_batch():
    cam = SimpleNamespace(world_to_camera=torch.eye(4), fx=torch.tensor(20.0), fy=torch.tensor(20.0),
                          cx=torch.tensor(20.0), cy=torch.tensor(20.0), width=torch.tensor(40), height=torch.tensor(40))
    return (cam, ("x", None), _inv())


class BehindWiringTest(unittest.TestCase):
    def _steps(self, behind):
        from internal.metrics.mcmc_citygsv2_metrics import MCMCCityGSV2Metrics
        m = MCMCCityGSV2Metrics(depth_support=True, depth_support_from_iter=0, depth_support_window=5,
                                depth_support_behind=behind).instantiate()
        model = _Model(MEANS.clone())
        gate = None
        for s in range(7):
            metrics, pbar = {"loss": torch.zeros(())}, {}
            gate = m._geom_gate_step(model, s, _cam_batch(), metrics, pbar, torch.sigmoid(model.opacities.reshape(-1)))
        return model, gate

    def test_off_sets_no_sampling_mask(self):
        model, gate = self._steps(False)
        self.assertFalse(hasattr(model, "_geom_gate_samp"))
        self.assertIs(model._geom_gate, gate)

    def test_behind_only_hits_sampling(self):
        model_off, gate_off = self._steps(False)
        model, gate = self._steps(True)
        self.assertTrue(torch.equal(gate, gate_off))                   # 熵／下壓力看的閘門不變
        self.assertTrue(bool(gate[1]))                                 # 後 2% 的那顆仍有深度支撐（|r| < 5%）
        self.assertEqual(model._geom_gate_samp.tolist(), (gate & torch.tensor([True, False, True, True, True])).tolist())

    def test_density_prefers_sampling_mask(self):
        from internal.density_controllers.mcmc_density_controller import MCMCDensityControllerImpl
        model, gate = self._steps(True)
        fake = SimpleNamespace(config=SimpleNamespace(geom_gate_sampling=True))
        p = MCMCDensityControllerImpl._geom_gate_probs(fake, model, torch.ones(5))
        self.assertAlmostEqual(float(p[1]), 1e-3, places=7)            # behind => 1e-3
        self.assertAlmostEqual(float(p[0]), 1 + 1e-3, places=6)
        idx = torch.tensor([1, 2])                                     # 搬遷路徑（只取活的）
        p2 = MCMCDensityControllerImpl._geom_gate_probs(fake, model, torch.ones(2), idx)
        self.assertAlmostEqual(float(p2[0]), 1e-3, places=7)
        # 沒開 behind => 讀 _geom_gate（與之前相同）
        model_off, gate_off = self._steps(False)
        p0 = MCMCDensityControllerImpl._geom_gate_probs(fake, model_off, torch.ones(5))
        self.assertTrue(torch.allclose(p0, gate_off.float() + 1e-3))


if __name__ == "__main__":
    unittest.main()
