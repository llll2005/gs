"""harvest_prune_dead（2026-11-07，使用者「直接設成預設」）：增生後每 k 步刪 o <= min_opacity 的死顆粒——純 CPU。"""
import types
import unittest

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl as C


class HarvestPruneDeadTest(unittest.TestCase):
    def setUp(self):
        self.pruned = []
        o = torch.tensor([[0.9], [0.004], [0.5], [0.001]])
        self.gm = types.SimpleNamespace(get_opacities=lambda: o)
        self.s = types.SimpleNamespace(config=types.SimpleNamespace(harvest_prune_dead_interval=500),
                                       _current_min_opacity=lambda step: 0.005,
                                       _prune_points=lambda m, gm, opt: self.pruned.append(m.clone()))

    def test_prunes_dead_on_interval(self):
        n = C._harvest_prune_dead(self.s, self.gm, None, 16000)
        self.assertEqual(n, 2)
        self.assertEqual(self.pruned[0].tolist(), [False, True, False, True])

    def test_skips_off_interval(self):
        self.assertEqual(C._harvest_prune_dead(self.s, self.gm, None, 16001), 0)
        self.assertEqual(self.pruned, [])


if __name__ == "__main__":
    unittest.main()
