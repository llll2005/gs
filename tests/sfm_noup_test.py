"""sfm_vis_noup（2026-11-06，使用者「讓沒證據的地方不准往上長」）：沒證據的顆粒 z 只降不升——純 CPU。"""
import types
import unittest

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl as C


class SfmNoUpTest(unittest.TestCase):
    def setUp(self):
        self.s = types.SimpleNamespace(config=types.SimpleNamespace(sfm_vis_noup_inherit=False, sfm_vis_noup_vote_every=0, densify_from_iter=1000))
        X = torch.tensor([[0., 0., 1.0], [0., 0., 2.0], [0., 0., 3.0]])
        # 第 0 顆：視野 5 台、有證據 0 => 標；第 1 顆：視野只有 1 台 => 不標；第 2 顆：有證據 => 不標
        C._sfm_noup_flag(self.s, X, torch.tensor([0, 0, 5]), torch.tensor([5, 1, 5]), 1050)
        self.gm = types.SimpleNamespace(gaussians={"means": X.clone()})

    def clamp(self):
        C._sfm_noup_clamp(self.s, None, None, self.gm, 0, None)
        return self.gm.gaussians["means"][:, 2].tolist()

    def test_only_no_evidence_is_flagged(self):
        z = self.s._noup_z
        self.assertEqual(float(z[0]), 1.0)
        self.assertTrue(torch.isinf(z[1:]).all())

    def test_cannot_move_up_but_can_move_down(self):
        self.gm.gaussians["means"][:, 2] += 0.5
        self.assertEqual(self.clamp(), [1.0, 2.5, 3.5])               # 只有被標的那顆被夾回
        self.gm.gaussians["means"][0, 2] = 0.2
        self.clamp()
        self.gm.gaussians["means"][0, 2] = 0.9                        # 往下之後再往上 => 夾回新的低點（只降不升）
        self.assertAlmostEqual(self.clamp()[0], 0.2, places=6)

    def test_released_when_evidence_appears(self):
        C._sfm_noup_flag(self.s, self.gm.gaussians["means"], torch.tensor([2, 0, 5]), torch.tensor([5, 1, 5]), 1150)
        self.assertTrue(torch.isinf(self.s._noup_z).all())

    def test_shape_mismatch_is_noop(self):
        self.gm.gaussians["means"] = torch.zeros(5, 3)                # 顆數變了（拓樸改變還沒同步）=> 不動
        self.assertEqual(self.clamp(), [0.0] * 5)


if __name__ == "__main__":
    unittest.main()
