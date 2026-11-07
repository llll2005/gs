"""SfM 視線一致性投票 sfm_vis_votes（2026-11-05）——純 CPU。"""
import unittest

import torch

from internal.density_controllers.mcmc_2dgs_density_controller import sfm_vis_votes


def _cam():
    # 相機在 (0,0,10) 往下看（x_cam = R x + T，COLMAP 慣例）：R = diag(1,-1,-1)、T = -R C = (0,0,10)
    R = torch.diag(torch.tensor([1.0, -1.0, -1.0]))[None]
    T = torch.tensor([[0.0, 0.0, 10.0]])
    one = lambda v: torch.tensor([float(v)])
    return R, T, one(100), one(100), one(50), one(50), one(100), one(100)


def _grid(empty=None):
    # 10 px 一格、10x10 格；SfM 平面 z=0 => 每格深度 10；empty＝沒有 SfM 的格 (row, col)
    d = torch.full((1, 10, 10), 10.0)
    if empty is not None:
        d[0, empty[0], empty[1]] = float("nan")
    return d, d.clone()


class SfmVisVotesTest(unittest.TestCase):
    def setUp(self):
        self.cam = _cam()

    def votes(self, X, empty=None, k_px=8.0):
        dmin, dmax = _grid(empty)
        return sfm_vis_votes(torch.tensor(X, dtype=torch.float32), *self.cam, dmin, dmax, 10, k_px)

    def test_on_surface_is_neither(self):
        f, b, c, _ = self.votes([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])   # 容差 8 px x 10 / 100 = 0.8
        self.assertEqual(f.tolist(), [0, 0]); self.assertEqual(b.tolist(), [0, 0]); self.assertEqual(c.tolist(), [1, 1])

    def test_below_surface_is_behind(self):
        f, b, c, _ = self.votes([[0.0, 0.0, -3.0]])
        self.assertEqual((int(f), int(b), int(c)), (0, 1, 1))

    def test_above_surface_is_free_space(self):
        f, b, c, _ = self.votes([[0.0, 0.0, 4.0]])
        self.assertEqual((int(f), int(b), int(c)), (1, 0, 1))

    def test_no_sfm_in_cell_is_no_evidence(self):
        # 投影到格 (0,0)：u=v=5 => x_cam=(5-50)*10/100=-4.5、y_cam=-4.5 => 世界 (-4.5, 4.5, 0)；放在平面上方也一樣沒證據
        f, b, c, iv = self.votes([[-4.5, 4.5, 0.0], [-2.25, 2.25, 5.0]], empty=(0, 0))
        self.assertEqual(c.tolist(), [0, 0]); self.assertEqual(f.tolist(), [0, 0]); self.assertEqual(b.tolist(), [0, 0])
        self.assertEqual(iv.tolist(), [1, 1])                           # 在視野內、只是沒有證據

    def test_outside_view_and_behind_camera(self):
        f, b, c, iv = self.votes([[100.0, 0.0, 0.0], [0.0, 0.0, 20.0]])
        self.assertEqual(c.tolist(), [0, 0]); self.assertEqual(iv.tolist(), [0, 0])

    def test_behind_max_drops_far_views(self):
        # ★ 11-08：容差＝8 px x 顆粒深度 / 100；behind_max 10。地下 3（深度 13、上限 10.4）仍算被擋；地下 60（深度 70、上限 56）整台不算證據
        dmin, dmax = _grid()
        X = torch.tensor([[0.0, 0.0, -3.0], [0.0, 0.0, -60.0]], dtype=torch.float32)
        f, b, c, iv, ca = sfm_vis_votes(X, *self.cam, dmin, dmax, 10, 8.0, behind_max=10.0, return_cover_all=True)
        self.assertEqual(b.tolist(), [1, 0]); self.assertEqual(c.tolist(), [1, 0]); self.assertEqual(ca.tolist(), [1, 1])
        f0, b0, c0, _ = sfm_vis_votes(X, *self.cam, dmin, dmax, 10, 8.0)            # 預設＝原行為（v4）
        self.assertEqual(b0.tolist(), [1, 1]); self.assertEqual(c0.tolist(), [1, 1])

    def test_tolerance_scales_with_pixels(self):
        X = [[0.0, 0.0, -0.5]]                                          # 0.5 在 8 px（0.8）內、在 2 px（0.2）外
        self.assertEqual(int(self.votes(X, k_px=8.0)[1][0]), 0)
        self.assertEqual(int(self.votes(X, k_px=2.0)[1][0]), 1)


if __name__ == "__main__":
    unittest.main()
