"""★ 2026-11-13 規格 5 §1：逐段計時依步型分帳（_StepProfiler.end_step(cls)／cls_report）——純 CPU（CUDA 呼叫換成假的）。"""
import unittest
from unittest import mock

import torch

import internal.gaussian_splatting as G


class _Ev:
    def __init__(self, *a, **k):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.0


def _patched():
    return mock.patch.multiple(torch.cuda, synchronize=lambda *a, **k: None, Event=_Ev,
                               max_memory_allocated=lambda *a, **k: 0, memory_reserved=lambda *a, **k: 0,
                               reset_peak_memory_stats=lambda *a, **k: None)


class StepProfileClsTest(unittest.TestCase):
    def _run(self, classes):
        with _patched(), mock.patch.dict("os.environ", {"CITYGS_STEP_PROFILE": "1"}):
            p = G._StepProfiler()
            p.WARMUP = 10 ** 9                       # 不重置
            for c in classes:
                p.start_step()
                p.mark("3 forward（光柵化）")
                p.mark("6 backward")
                p.end_step(c)
                p._acc("11 週期性 trim pass（after_training_step）", 1.0, 5.0)    # 收尾段：記到同一步的步型
            return p

    def test_counts_and_tail_attribution(self):
        p = self._run(["lean", "dist", "lean", "full＋增生事件", "lean"])
        self.assertEqual(p.cls_n, {"lean": 3, "dist": 1, "full＋增生事件": 1})
        self.assertEqual(sum(p.cls_n.values()), p.steps)
        self.assertAlmostEqual(p.cls_seg["lean"]["11 週期性 trim pass（after_training_step）"], 15.0)
        self.assertAlmostEqual(p.cls_seg["dist"]["11 週期性 trim pass（after_training_step）"], 5.0)
        # 迴圈外：第一步沒有（_prev_end 還沒設）；之後每步都記到「這一步」的步型
        self.assertIn("0 迴圈外（dataloader/Lightning）", p.cls_seg["dist"])
        self.assertAlmostEqual(sum(p.cls_real.values()), p.real_step_ms, places=6)
        rep = "\n".join(p.cls_report())
        for c in ("lean", "dist", "full＋增生事件", "（全部）"):
            self.assertIn(c, rep)

    def test_off_does_nothing(self):
        with mock.patch.dict("os.environ", {"CITYGS_STEP_PROFILE": ""}):
            p = G._StepProfiler()
            p.start_step(); p.end_step("lean"); p.sub_acc("x", 1.0)
            self.assertEqual((p.steps, p.cls_n, p.sub), (0, {}, {}))
            self.assertEqual(p.cls_report(), [])

    def test_warmup_resets_classes(self):
        with _patched(), mock.patch.dict("os.environ", {"CITYGS_STEP_PROFILE": "1"}):
            p = G._StepProfiler()
            p.WARMUP = 2
            for c in ("full", "full", "lean"):
                p.start_step(); p.mark("3 forward（光柵化）"); p.end_step(c)
                p.sub_acc("hooks", 1.0)
            self.assertEqual(p.cls_n, {"lean": 1})                 # 暖機兩步被歸零
            self.assertEqual(p.sub, {"hooks": [2.0, 2]})          # 第 2 步的 hook 跑在重置之後（同一步的收尾）=> 算進穩態


if __name__ == "__main__":
    unittest.main()
