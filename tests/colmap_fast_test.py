"""★ 11-14 demo 模式 A：快速 COLMAP 讀法與原本讀法逐值相同（用真資料；沒有資料就跳過）。
python -m unittest tests.colmap_fast_test"""
import os
import tempfile
import shutil
import unittest

import numpy as np

SP = "data/matrix_city/aerial/train/block_all/sparse/0"


@unittest.skipUnless(os.path.exists(os.path.join(SP, "points3D.bin")), "缺 MatrixCity sparse/0")
class ColmapFastTest(unittest.TestCase):
    def test_images_light_same_poses(self):
        from internal.utils.colmap import read_images_binary
        from internal.utils.colmap_fast import read_images_binary_light
        a = read_images_binary(os.path.join(SP, "images.bin"))
        b = read_images_binary_light(os.path.join(SP, "images.bin"))
        self.assertEqual(list(a.keys()), list(b.keys()))
        for k in a:
            self.assertEqual(a[k].name, b[k].name)
            self.assertEqual(a[k].camera_id, b[k].camera_id)
            self.assertTrue(np.array_equal(a[k].qvec, b[k].qvec))
            self.assertTrue(np.array_equal(a[k].tvec, b[k].tvec))

    def test_points3D_same_with_and_without_filter(self):
        from internal.dataparsers.colmap_dataparser import ColmapDataParser
        from internal.utils import colmap_fast
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "points3D.bin")
            shutil.copy(os.path.join(SP, "points3D.bin"), p)       # 快取寫在暫存目錄，不動資料夾
            os.environ.pop("CITYGS_SFM_FAST", None)
            ref = ColmapDataParser.read_points3D_binary(p)
            rng = np.random.default_rng(0)
            sel = {int(i): True for i in rng.choice(np.arange(1, 5622), 300, replace=False)}
            ref_sel = ColmapDataParser.read_points3D_binary(p, selected_image_ids=sel)
            os.environ["CITYGS_SFM_FAST"] = "1"
            for _ in range(2):                                        # 第一次解析＋寫快取，第二次讀快取
                got = ColmapDataParser.read_points3D_binary(p)
                got_sel = ColmapDataParser.read_points3D_binary(p, selected_image_ids=sel)
                for r, g in zip(ref, got):
                    self.assertEqual(r.dtype, g.dtype); self.assertTrue(np.array_equal(r, g))
                for r, g in zip(ref_sel, got_sel):
                    self.assertEqual(r.dtype, g.dtype); self.assertTrue(np.array_equal(r, g))
            self.assertTrue(os.path.exists(p + ".cache.npz"))
            self.assertLess(len(ref_sel[0]), len(ref[0]))
        finally:
            os.environ.pop("CITYGS_SFM_FAST", None)
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
