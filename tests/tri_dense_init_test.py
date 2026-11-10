"""★ 2026-11-11 稠密初始化的體素支撐（tools/make_tri_dense_init.py）。"""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from make_tri_dense_init import supported_voxels, count_supported, solve_voxel, voxel_keys   # noqa: E402


class DenseInitTest(unittest.TestCase):
    def _scene(self, n_views=4, per=4000, seed=0):
        rng = np.random.default_rng(seed)
        X, V = [], []
        for v in range(n_views):                       # 每個視角在 z=0 平面上撒點（多視角一致）
            p = rng.uniform(0, 10, (per, 3)); p[:, 2] = rng.normal(0, 0.005, per)
            X.append(p); V.append(np.full(per, v))
        out = rng.uniform(0, 10, (500, 3)); out[:, 2] = rng.uniform(1, 9, 500)   # 單一視角的離群點（離開平面）
        X.append(out); V.append(np.full(500, 0))
        X = np.concatenate(X); V = np.concatenate(V).astype(np.int32)
        C = np.full((len(X), 3), 100, np.uint8)
        return X, C, V

    def test_single_view_outliers_removed(self):
        X, C, V = self._scene()
        cen, col = supported_voxels(X, C, V, np.array([-1e-3, -1e-3, -1.0]), 0.5, 2)
        self.assertGreater(len(cen), 300)
        self.assertLess(np.abs(cen[:, 2]).max(), 0.5)          # 留下的都在平面那一層
        self.assertTrue(np.allclose(col, 100))
        cen1, _ = supported_voxels(X, C, V, np.array([-1e-3, -1e-3, -1.0]), 0.5, 1)
        self.assertGreater(np.abs(cen1[:, 2]).max(), 1.0)       # 不要求多視角時離群點會留下

    def test_blocked_voxels_skipped(self):
        X, C, V = self._scene()
        o = np.array([-1e-3, -1e-3, -1.0])
        base = np.array([[0.25, 0.25, 0.0], [5.25, 5.25, 0.0]])
        n0 = count_supported(X, V, o, 0.5, 2)
        n1 = count_supported(X, V, o, 0.5, 2, base)
        self.assertEqual(n0 - n1, 2)
        cen, _ = supported_voxels(X, C, V, o, 0.5, 2, np.unique(voxel_keys(base, o, 0.5)))
        self.assertEqual(len(cen), n1)

    def test_solver_hits_target_on_supported_branch(self):
        X, C, V = self._scene(n_views=6, per=20000)
        o = np.array([-1e-3, -1e-3, -1.0])
        g, n = solve_voxel(X, V, o, 1500, 2, None, 10.0 / 4)
        self.assertGreaterEqual(n, 1500)
        self.assertLess(n, 1500 * 1.15)
        self.assertGreater(g, 0.1)                              # 沒有掉到「每格湊不滿兩個視角」那一支
        # 要得比候選撐得起的還多 => 回傳格數最多的那個
        g2, n2 = solve_voxel(X, V, o, 10 ** 7, 2, None, 10.0 / 4)
        self.assertLess(n2, 10 ** 7)
        self.assertGreater(n2, 5000)

    def test_solver_scans_past_coarse_cells_blocked_by_base(self):
        """11-11 lab 第一次跑：粗格全被 SfM＋補點佔滿（格數 0～1），掃描誤判「過了峰值」提早停 => 只加 1 顆。"""
        X, C, V = self._scene(n_views=6, per=20000)
        o = np.array([-1e-3, -1e-3, -1.0])
        rng = np.random.default_rng(5)
        base = rng.uniform(0, 10, (60, 3)); base[:, 2] = 0.0          # 稀疏的 SfM 點：粗格（邊長 2.5）每格都有、細格大多沒有
        self.assertLessEqual(count_supported(X, V, o, 2.5, 2, base), 1)
        g, n = solve_voxel(X, V, o, 1500, 2, base, 2.5)
        self.assertGreaterEqual(n, 1500)
        self.assertLess(g, 1.0)

if __name__ == "__main__":
    unittest.main()
