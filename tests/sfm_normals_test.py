import unittest
from types import SimpleNamespace

import numpy as np
import torch

from internal.density_controllers.sfm_normals import SfMNormals, quat_with_normal, snap_rotations
from internal.density_controllers.merge_ops import _qmat


def _grid_plane(axis: int, n=30, spacing=1.0):
    a, b = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing, indexing="ij")
    P = np.zeros((n * n, 3)); o = [i for i in range(3) if i != axis]
    P[:, o[0]], P[:, o[1]] = a.ravel(), b.ravel()
    return P


class SfMNormalsTest(unittest.TestCase):
    def test_plane_normals_and_validity(self):
        P = np.concatenate([_grid_plane(2), _grid_plane(0) + [50, 0, 0]])     # 地面 z=0、牆 x=50
        sn = SfMNormals(P, k=8)
        n, ok = sn.query(np.array([[10.2, 10.3, 0.1], [50.1, 10.0, 10.0], [10.0, 10.0, 20.0]]))
        self.assertTrue(ok[0] and ok[1])
        self.assertGreater(abs(n[0, 2]), 0.99)
        self.assertGreater(abs(n[1, 0]), 0.99)
        self.assertFalse(ok[2])                                                # 離最近 SfM 點超過它的近鄰半徑

    def test_scale_free(self):
        P = _grid_plane(2)
        q = np.array([[10.2, 10.3, 0.1], [10.0, 10.0, 5.0]])
        n1, ok1 = SfMNormals(P, k=8).query(q)
        n2, ok2 = SfMNormals(P * 0.01, k=8).query(q * 0.01)
        self.assertTrue((ok1 == ok2).all())
        self.assertTrue(np.allclose(np.abs(n1), np.abs(n2), atol=1e-5))

    def test_quat_with_normal(self):
        g = torch.Generator().manual_seed(0)
        q0 = torch.randn(100, 4, generator=g)
        n = torch.nn.functional.normalize(torch.randn(100, 3, generator=g), dim=-1)
        R = _qmat(quat_with_normal(n, q0))
        self.assertTrue(torch.allclose(R[:, :, 2], n, atol=1e-4))
        self.assertTrue(torch.allclose(R.transpose(1, 2) @ R, torch.eye(3).expand(100, 3, 3), atol=1e-4))

    def test_snap_rotations_only_valid(self):
        sn = SfMNormals(_grid_plane(2), k=8)
        gm = SimpleNamespace(gaussians={"means": torch.tensor([[10.2, 10.3, 0.1], [10.0, 10.0, 20.0]]),
                                        "rotations": torch.tensor([[0.7071, 0.7071, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]])})
        r1 = gm.gaussians["rotations"][1].clone()
        st = snap_rotations(gm, sn)
        self.assertEqual(st["ok"], 1)
        self.assertGreater(abs(float(_qmat(gm.gaussians["rotations"][:1])[0, 2, 2])), 0.999)
        self.assertTrue(torch.equal(gm.gaussians["rotations"][1], r1))


if __name__ == "__main__":
    unittest.main()
