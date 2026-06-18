import unittest

import torch

from metrics import compare_logits


class CompareLogitsTests(unittest.TestCase):
    def test_identical_logits_have_no_effect(self) -> None:
        logits = torch.tensor([1.0, 2.0, 3.0])
        effect = compare_logits(logits, logits)
        self.assertEqual(effect.l2, 0.0)
        self.assertEqual(effect.max_abs, 0.0)
        self.assertEqual(effect.kl_divergence, 0.0)

    def test_changed_logits_have_positive_effect(self) -> None:
        effect = compare_logits(torch.zeros(2), torch.tensor([1.0, -1.0]))
        self.assertGreater(effect.l2, 0)
        self.assertGreater(effect.max_abs, 0)
        self.assertGreater(effect.kl_divergence, 0)

    def test_rejects_different_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes"):
            compare_logits(torch.zeros(2), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
