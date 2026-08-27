import math
import unittest

import numpy as np
import torch

import sentiment_data as data
from experiments.phase7_sentiment import make_split
from experiments.phase8_persistence import lexicon_score, write_layers_for
from relay import CalibratedReader, LatentWriter
from stats import auroc, bootstrap_ci, sign_flip_p, spearman


class DataTests(unittest.TestCase):
    def test_classes_are_balanced_and_unique(self):
        self.assertEqual(len(data.EXPLICIT_POSITIVE), len(data.EXPLICIT_NEGATIVE))
        self.assertEqual(len(data.IMPLICIT_POSITIVE), len(data.IMPLICIT_NEGATIVE))
        everything = (data.EXPLICIT_POSITIVE + data.EXPLICIT_NEGATIVE
                      + data.IMPLICIT_POSITIVE + data.IMPLICIT_NEGATIVE)
        self.assertEqual(len(everything), len(set(everything)))

    def test_dev_and_test_prompts_do_not_overlap(self):
        self.assertFalse(set(data.DEV_TARGETS) & set(data.TEST_TARGETS))
        self.assertGreaterEqual(len(data.TEST_TARGETS), 30)

    def test_word_lists_do_not_overlap(self):
        self.assertFalse(set(data.POSITIVE_WORDS) & set(data.NEGATIVE_WORDS))

    def test_neutral_groups_are_separate(self):
        groups = [data.NEUTRAL_CENTER, data.NEUTRAL_TURNS, data.NEUTRAL_DEV, data.NEUTRAL_REPORT]
        flat = [sentence for group in groups for sentence in group]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertGreaterEqual(len(data.NEUTRAL_REPORT), 20)
        self.assertGreaterEqual(len(data.NEUTRAL_TURNS), 5)

    def test_split_is_disjoint_and_complete(self):
        fit, held = make_split(48, 3)
        self.assertEqual(sorted([*fit, *held]), list(range(48)))
        self.assertEqual(len(fit), len(held))

    def test_lexicon_score_sign(self):
        self.assertGreater(lexicon_score("what a wonderful and happy day"), 0)
        self.assertLess(lexicon_score("a terrible and sad day"), 0)
        self.assertEqual(lexicon_score("the door is open"), 0)

    def test_write_layers_stay_in_range(self):
        for count in (12, 24):
            layers = write_layers_for(count)
            self.assertTrue(all(1 <= layer <= count for layer in layers))
            self.assertEqual(layers, sorted(set(layers)))


class ReaderWriterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        shift = torch.zeros(16)
        shift[0] = 3.0
        offset = torch.full((16,), 50.0)  # large shared component, like real activations
        self.pos = torch.randn(30, 16) * 0.5 + offset + shift
        self.neg = torch.randn(30, 16) * 0.5 + offset - shift

    def test_midpoint_is_zero_and_classes_split_by_sign(self):
        reader = CalibratedReader.from_examples(self.pos, self.neg)
        self.assertTrue((reader.read_state(self.pos) > 0).all())
        self.assertTrue((reader.read_state(self.neg) < 0).all())
        midpoint = (self.pos.mean(0) + self.neg.mean(0)) / 2
        self.assertAlmostEqual(float(reader.read_state(midpoint)), 0.0, places=5)

    def test_neutral_zero_point(self):
        neutral = torch.randn(10, 16) * 0.5 + 50.0 + torch.tensor([1.0] + [0.0] * 15)
        reader = CalibratedReader.from_examples(self.pos, self.neg, neutral)
        self.assertAlmostEqual(float(reader.read_state(neutral.mean(0))), 0.0, places=5)
        self.assertGreater(float(reader.read_state(self.pos).mean()), 0)
        self.assertLess(float(reader.read_state(self.neg).mean()), 0)
        with self.assertRaises(ValueError):
            CalibratedReader.from_examples(self.pos, self.neg, torch.zeros(3, 8))

    def test_state_is_bounded_and_keeps_batch_shape(self):
        reader = CalibratedReader.from_examples(self.pos, self.neg)
        z = reader.read_state(torch.stack((self.pos, self.neg)) * 100)
        self.assertEqual(z.shape, (2, 30))
        self.assertTrue((z.abs() <= 1).all())

    def test_wrong_size_rejected(self):
        reader = CalibratedReader.from_examples(self.pos, self.neg)
        with self.assertRaises(ValueError):
            reader.read_state(torch.zeros(8))

    def test_writer_from_direction_normalizes_and_shifts(self):
        writer = LatentWriter.from_direction(torch.tensor([3.0, 4.0]))
        self.assertAlmostEqual(float(writer.direction.norm()), 1.0, places=6)
        out = writer.write_state(torch.zeros(1, 2, 2), 0.5, alpha=2.0)
        self.assertTrue(torch.allclose(out[0, 0], torch.tensor([0.6, 0.8])))

    def test_writer_rejects_bad_direction(self):
        for bad in (torch.zeros(4), torch.tensor([float("nan"), 1.0]), torch.zeros(2, 2)):
            with self.assertRaises(ValueError):
                LatentWriter.from_direction(bad)


class StatsTests(unittest.TestCase):
    def test_auroc_extremes_and_ties(self):
        self.assertEqual(auroc([3, 4, 1, 2], [True, True, False, False]), 1.0)
        self.assertEqual(auroc([1, 2, 3, 4], [True, True, False, False]), 0.0)
        self.assertEqual(auroc([1, 1, 1, 1], [True, True, False, False]), 0.5)

    def test_bootstrap_ci_contains_mean(self):
        mean, low, high = bootstrap_ci(np.arange(20.0))
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_sign_flip_detects_effect_not_noise(self):
        self.assertLess(sign_flip_p(np.full(20, 1.0) + np.random.default_rng(0).normal(0, 0.1, 20)), 0.01)
        self.assertGreater(sign_flip_p(np.array([1.0, -1.0] * 10)), 0.5)

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(math.isfinite(spearman([1, 2, 3], [5, 5, 5])))


if __name__ == "__main__":
    unittest.main()
