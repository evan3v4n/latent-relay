import unittest

import torch

from relay import LatentState, fit_direction
from trainable import TrainableRelay


class TrainableTests(unittest.TestCase):
    def test_vector_broadcast_is_per_batch(self):
        relay = TrainableRelay(8, 4)
        z = torch.randn(2, 4)
        hidden = torch.zeros(2, 3, 8)
        output = relay.write_state(hidden, z)
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.equal(output[:, 0], relay.write(z)))
        self.assertTrue(torch.equal(output[:, 0], output[:, 2]))

    def test_scalar_and_vector_shapes(self):
        for size in (1, 4, 8, 16):
            relay = TrainableRelay(8, size)
            observed = relay.read_state(torch.randn(8))
            state = LatentState(0.5, size)
            stored = state.update(observed)
            result = relay.write_state(torch.zeros(1, 3, 8), stored)
            self.assertEqual(result.shape, (1, 3, 8))
            self.assertTrue(torch.all(torch.as_tensor(stored).abs() <= 0.5))

    def test_invalid_state_does_not_corrupt_memory(self):
        state = LatentState()
        state.update(0.5)
        for invalid in (float("nan"), float("inf"), 1.01, torch.ones(2)):
            with self.assertRaises(ValueError):
                state.update(invalid)
            self.assertEqual(state.z, 0.5)

    def test_empty_or_nonfinite_examples_rejected(self):
        for invalid in (torch.empty(0, 3), torch.full((2, 3), float("nan"))):
            with self.assertRaises(ValueError):
                fit_direction(invalid, torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
