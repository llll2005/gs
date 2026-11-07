"""px_floor（2026-11-07，點精靈吸收態的修法）：尺度夾在「最近可見相機裡 px_floor 像素」以上——純 CPU。"""
import math
import types
import unittest

import torch

from internal.cameras.cameras import Cameras
from internal.density_controllers.mcmc_2dgs_density_controller import MCMC2DGSDensityControllerImpl as C


def _cams():
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    return Cameras(R=torch.eye(3)[None].repeat(2, 1, 1), T=t([[0., 0., 0.], [0., 0., 5.]]), fx=t([100., 100.]), fy=t([100., 100.]),
                   cx=t([50., 50.]), cy=t([50., 50.]), width=torch.tensor([100, 100], dtype=torch.int16), height=torch.tensor([100, 100], dtype=torch.int16),
                   appearance_id=torch.zeros(2, dtype=torch.int), normalized_appearance_id=torch.zeros(2), distortion_params=None,
                   camera_type=torch.zeros(2, dtype=torch.int8))


class PxFloorTest(unittest.TestCase):
    def setUp(self):
        # 相機 0 在原點看 +z；相機 1 往後 5（深度 +5）。第 0 顆：兩台都看得到，最近深度 10；第 1 顆：畫面外；第 2 顆：在兩台後面
        self.X = torch.tensor([[0., 0., 10.], [100., 0., 10.], [0., 0., -20.]])

    def test_floor_is_nearest_visible_depth_in_pixels(self):
        fl = C._px_floor_log(self.X, _cams(), 1.0, 1.0)
        self.assertAlmostEqual(math.exp(float(fl[0])), 10. / 100., places=6)
        self.assertTrue(torch.isneginf(fl[1:]).all())

    def test_lowres_doubles_floor(self):
        fl = C._px_floor_log(self.X, _cams(), 0.5, 1.0)
        self.assertAlmostEqual(math.exp(float(fl[0])), 0.2, places=6)

    def test_clamp_raises_only_small_scales(self):
        s = types.SimpleNamespace(config=types.SimpleNamespace(px_floor=1.0, px_floor_interval=50), _px_floor_log=C._px_floor_log)
        sc = torch.log(torch.tensor([[0.01, 0.02], [1e-6, 1e-6], [1e-6, 1e-6], [0.01, 1.0]]))
        self.X = torch.cat([self.X, self.X[:1]])                                  # 第 3 顆＝第 0 顆位置的細長筆觸
        gm = types.SimpleNamespace(gaussians={"means": self.X.clone(), "scales": sc})
        pl = types.SimpleNamespace(_res_now=1.0, trainer=types.SimpleNamespace(datamodule=types.SimpleNamespace(
            dataparser_outputs=types.SimpleNamespace(train_set=types.SimpleNamespace(cameras=_cams())))))
        C._px_floor_clamp(s, None, None, gm, 7, pl)
        e = torch.exp(gm.gaussians["scales"])
        self.assertTrue(torch.allclose(e[0], torch.tensor([0.1, 0.1])))         # 長軸 0.02 < 下限 0.1 => 兩軸都拉到 0.1
        self.assertTrue(torch.allclose(e[1:3], torch.full((2, 2), 1e-6)))       # 沒有相機看得到 => 不夾
        self.assertTrue(torch.allclose(e[3], torch.tensor([0.01, 1.0])))        # 細長筆觸（長軸夠長）=> 不動

    def test_new_particles_get_no_floor_until_inherited(self):
        s = types.SimpleNamespace(_pxf_log=torch.tensor([0.5, -1.0]), _PER_PARTICLE=C._PER_PARTICLE)
        C._per_particle_extend(s, 2, 4)                                          # 新增 2 顆 => 先補 -inf（不夾），add_new_gs 再繼承母點
        self.assertEqual(s._pxf_log.shape[0], 4)
        self.assertTrue(torch.isneginf(s._pxf_log[2:]).all())


if __name__ == "__main__":
    unittest.main()
