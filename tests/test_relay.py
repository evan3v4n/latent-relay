import unittest

import torch

from relay import LatentReader


class LatentReaderTests(unittest.TestCase):
    def test_direction_is_unit_length_and_reproducible(self) -> None:
        first = LatentReader(8, seed=7)
        second = LatentReader(8, seed=7)
        self.assertTrue(torch.allclose(first.direction.norm(), torch.tensor(1.0)))
        self.assertTrue(torch.equal(first.direction, second.direction))

    def test_reads_scalar_and_batched_states(self) -> None:
        reader = LatentReader(4)
        self.assertEqual(reader.read_state(torch.ones(4)).shape, torch.Size([]))
        values = reader.read_state(torch.randn(3, 4))
        self.assertEqual(values.shape, torch.Size([3]))
        self.assertTrue(torch.all(values.abs() <= 1))

    def test_rejects_wrong_hidden_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected hidden size 4"):
            LatentReader(4).read_state(torch.ones(3))


if __name__ == "__main__":
    unittest.main()
