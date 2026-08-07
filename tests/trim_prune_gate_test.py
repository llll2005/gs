"""The trim schedule must be expressible independently of the densify schedule.

`after_training_step` used to gate trimming on `density_controller.config.densify_until_iter`,
which makes "stop densifying early but keep harvesting" impossible to configure -- the very
schedule 紀錄 lists as an open experiment. `contribution_prune_until_iter` separates them;
-1 keeps the shipped coupling so every existing config is unaffected.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer


def _module(densify_until):
    """The smallest object `after_training_step` will accept, with 0 cameras."""
    pruned = []
    return SimpleNamespace(
        trainer=SimpleNamespace(datamodule=SimpleNamespace(
            dataparser_outputs=SimpleNamespace(train_set=SimpleNamespace(cameras=[])))),
        gaussian_model=SimpleNamespace(get_xyz=torch.zeros(1, 3)),
        gaussian_optimizers=None,
        density_controller=SimpleNamespace(
            config=SimpleNamespace(densify_until_iter=densify_until),
            _prune_points=lambda m, g, o: pruned.append(int(m.sum())),
        ),
        _fixed_background_color=lambda: torch.zeros(3),
    ), pruned


def _ran(renderer, step, densify_until):
    """True if the trim body executed at `step`."""
    module, pruned = _module(densify_until)
    # 0 cameras -> the accumulator has nothing to reduce, so stand in a fixed contribution vector.
    acc = (lambda out: None, lambda: torch.tensor([0.0, 1.0, 2.0, 3.0]))
    with mock.patch(
            "internal.renderers.sep_depth_trim_2dgs_renderer.contribution_accumulator",
            return_value=acc):
        renderer.after_training_step(step, module)
    return len(pruned) == 1


class TrimPruneGateTest(unittest.TestCase):
    def _renderer(self, **kw):
        kw.setdefault("contribution_prune_from_iter", 1000)
        kw.setdefault("contribution_prune_interval", 500)
        return SepDepthTrim2DGSRenderer(**kw)

    def test_default_follows_densify_until_iter(self):
        """-1 must reproduce the old behaviour exactly, or every existing run changes."""
        r = self._renderer()
        self.assertTrue(_ran(r, 15_000, densify_until=16_000))
        self.assertFalse(_ran(r, 16_500, densify_until=16_000))

    def test_explicit_value_outlives_densify(self):
        """The point of the flag: harvest for 8k steps after densify has stopped."""
        r = self._renderer(contribution_prune_until_iter=24_000)
        self.assertTrue(_ran(r, 20_000, densify_until=16_000))

    def test_explicit_value_can_also_stop_early(self):
        """It is a real bound, not just a way to extend."""
        r = self._renderer(contribution_prune_until_iter=10_000)
        self.assertFalse(_ran(r, 15_000, densify_until=16_000))

    def test_other_gates_still_apply(self):
        """from_iter and interval must keep working under the new bound."""
        r = self._renderer(contribution_prune_until_iter=24_000)
        self.assertFalse(_ran(r, 500, densify_until=16_000), "before from_iter")
        self.assertFalse(_ran(r, 20_001, densify_until=16_000), "off-interval")

    def test_disable_wins(self):
        r = self._renderer(contribution_prune_until_iter=24_000, diable_trimming=True)
        self.assertFalse(_ran(r, 20_000, densify_until=16_000))




class TrimBlurScoreTest(unittest.TestCase):
    """The trim pass now unpacks a (mean, covered) tuple and stashes a blur score.

    The mocked tests above run with 0 cameras, so they never execute the loop body -- exactly the
    hole that let a `prog_bar[name]` KeyError reach a real run earlier today. This one puts a
    camera in and patches the render call, so the unpacking and the stash are actually exercised.
    """

    def test_blur_score_is_stashed_on_the_controller(self):
        r = SepDepthTrim2DGSRenderer(contribution_prune_from_iter=1000,
                                     contribution_prune_interval=500)
        module, _ = _module(densify_until=16_000)
        module.trainer.datamodule.dataparser_outputs.train_set.cameras = [
            SimpleNamespace(to_device=lambda d: None)]
        mean = torch.tensor([0.10, 0.20, 0.30, 0.40])
        covered = torch.tensor([10, 20, 30, 40])
        acc = (lambda out: None, lambda: torch.tensor([0.0, 1.0, 2.0, 3.0]))
        with mock.patch(
                "internal.renderers.sep_depth_trim_2dgs_renderer.contribution_accumulator",
                return_value=acc), \
             mock.patch.object(SepDepthTrim2DGSRenderer, "__call__",
                               return_value=(mean, covered)):
            r.after_training_step(15_000, module)
        score = module.density_controller.blur_score
        self.assertIsNotNone(score, "blur_score 沒有被寫入")
        # one camera, so the per-view mean is mean*covered itself
        self.assertTrue(torch.allclose(score, mean * covered.float()))


if __name__ == "__main__":
    unittest.main()
