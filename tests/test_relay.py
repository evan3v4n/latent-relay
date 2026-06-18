import unittest

import torch

from relay import LatentReader, LatentState, LatentWriter, fit_direction


class FitDirectionTests(unittest.TestCase):
    def test_fits_normalized_difference(self) -> None:
        positive = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        negative = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        direction = fit_direction(positive, negative)
        self.assertTrue(torch.allclose(direction, torch.tensor([1.0, -1.0]) / 2**0.5))

    def test_rejects_zero_difference(self) -> None:
        with self.assertRaisesRegex(ValueError, "no direction"):
            fit_direction(torch.ones(2, 3), torch.ones(2, 3))


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

    def test_read_is_independent_of_activation_magnitude(self) -> None:
        reader = LatentReader(4)
        hidden = torch.randn(4)
        self.assertTrue(
            torch.allclose(reader.read_state(hidden), reader.read_state(hidden * 100))
        )

    def test_rejects_wrong_hidden_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected hidden size 4"):
            LatentReader(4).read_state(torch.ones(3))

    def test_can_use_supplied_direction(self) -> None:
        reader = LatentReader.from_direction(torch.tensor([3.0, 0.0]))
        self.assertAlmostEqual(
            reader.read_state(torch.tensor([1.0, 0.0])).item(), 0.761594, places=6
        )


class LatentStateTests(unittest.TestCase):
    def test_updates_and_resets_persistent_state(self) -> None:
        state = LatentState(beta=0.5)
        self.assertEqual(state.update(1.0), 0.5)
        self.assertEqual(state.update(-0.5), 0.0)
        state.reset()
        self.assertEqual(state.z, 0.0)

    def test_rejects_invalid_beta(self) -> None:
        with self.assertRaisesRegex(ValueError, "beta"):
            LatentState(1.1)


class LatentWriterTests(unittest.TestCase):
    def test_writes_expected_unit_direction(self) -> None:
        writer = LatentWriter(4, seed=3)
        hidden = torch.zeros(2, 3, 4)
        written = writer.write_state(hidden, z=0.5, alpha=2.0)
        self.assertTrue(torch.allclose(written, writer.direction.expand_as(hidden)))
        self.assertTrue(torch.allclose(writer.direction.norm(), torch.tensor(1.0)))

    def test_zero_state_does_not_change_activation(self) -> None:
        writer = LatentWriter(4)
        hidden = torch.randn(2, 4)
        self.assertTrue(torch.equal(writer.write_state(hidden, z=0), hidden))


if __name__ == "__main__":
    unittest.main()
