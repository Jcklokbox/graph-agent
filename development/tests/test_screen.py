import copy
import json
import os
import tempfile
import unittest

from eval.screen import Screen, replay

BASE = os.path.join(os.path.dirname(__file__), "..", "results", "dev16_v11.json")


class ScreenRules(unittest.TestCase):
    def _replay(self, shift):
        base = json.load(open(BASE))
        cand = copy.deepcopy(base)
        for v in cand["variants"].values():
            for r in v["raw"]:
                r["brain"] += shift * (r["oracle"] - r["stay"])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cand, f)
        try:
            return replay(f.name, Screen(base, list(base["variants"]), .045, 16), 2)
        finally:
            os.unlink(f.name)

    def test_identical_is_no_gain(self):
        self.assertEqual(self._replay(0.)[0], "NO-GAIN")

    def test_uniform_loss_stops_early(self):
        verdict, _, t = self._replay(-.2)
        self.assertIn(verdict, ("REGRESSION", "WORSE"))
        self.assertLess(t, 10 * 60)

    def test_uniform_gain_is_better(self):
        self.assertEqual(self._replay(+.2)[0], "BETTER")


if __name__ == "__main__":
    unittest.main()
