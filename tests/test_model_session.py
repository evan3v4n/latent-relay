"""Real transformer integration tests with random tiny Pythia weights; no downloads."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from model import FrozenTransformer
from session import LatentRelay
from trainable import TrainableRelay, load_checkpoint, save_checkpoint


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def convert_ids_to_tokens(self, ids):
        return [str(value) for value in ids]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def tiny_transformer():
    wrapper = FrozenTransformer.__new__(FrozenTransformer)
    wrapper.device = torch.device("cpu")
    wrapper.model_name = "tiny-test"
    wrapper.tokenizer = TinyTokenizer()
    with torch.random.fork_rng():
        torch.manual_seed(0)
        wrapper.model = GPTNeoXForCausalLM(GPTNeoXConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=32, rotary_pct=0.5,
            attention_dropout=0.0, hidden_dropout=0.0,
        ))
    wrapper.model.eval().requires_grad_(False)
    return wrapper


class ModelSessionTests(unittest.TestCase):
    def setUp(self):
        self.transformer = tiny_transformer()

    def test_zero_state_identity_and_nonzero_effect(self):
        relay = LatentRelay(self.transformer, 1, 1)
        plain = self.transformer.next_token_logits("target")
        self.assertTrue(torch.equal(plain, relay.next_token_logits("target")))
        relay.remember("source")
        self.assertNotEqual(relay.z, 0)
        self.assertFalse(torch.equal(plain, relay.next_token_logits("target")))
        relay.reset()
        self.assertTrue(torch.equal(plain, relay.next_token_logits("target")))

    def test_hook_removed_on_transform_failure(self):
        block = self.transformer.model.gpt_neox.layers[0]
        def fail(hidden):
            raise RuntimeError("intentional failure")
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            self.transformer.next_token_logits("prompt", 1, fail)
        self.assertEqual(len(block._forward_hooks), 0)
        self.assertTrue(torch.isfinite(self.transformer.next_token_logits("prompt")).all())

    def test_training_gradient_reaches_maps_and_update_only(self):
        relay = TrainableRelay(16, 4)
        hidden = self.transformer.trace("source", [1]).last_token_by_layer[1].clone()
        hooks_before = len(self.transformer.model.gpt_neox.layers[0]._forward_hooks)
        observed = relay.read_state(hidden)
        z = relay.update_state(torch.zeros_like(observed), observed)
        logits = self.transformer.differentiable_logits(
            self.transformer.encode("target"), 1, lambda h: relay.write_state(h, z)
        )
        torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([4])).backward()
        for name, parameter in relay.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.transformer.model.parameters()))
        self.assertEqual(len(self.transformer.model.gpt_neox.layers[0]._forward_hooks), hooks_before)

    def test_generation_preserves_token_ids_and_stops_at_eos(self):
        relay = LatentRelay(self.transformer, 1, 1)
        seen = []
        def logits(ids, *args):
            seen.append(ids.tolist())
            result = torch.zeros(16)
            result[5 if len(seen) == 1 else 0] = 10
            return result
        with patch.object(self.transformer, "differentiable_logits", side_effect=logits):
            self.assertEqual(relay.generate("prompt", max_new_tokens=4), "1 2 3 5 0")
        self.assertEqual(seen, [[[1, 2, 3]], [[1, 2, 3, 5]]])

    def test_checkpoint_round_trip_and_vector_session(self):
        module = TrainableRelay(16, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relay.pt"
            save_checkpoint(path, module, model_name="tiny-test", read_layer=1, write_layer=2, alpha=2.0)
            restored, metadata = load_checkpoint(path)
            self.assertEqual(metadata["latent_size"], 4)
            for first, second in zip(module.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(first, second))
            session = LatentRelay.from_checkpoint(self.transformer, path)
        observation = session.remember("source")
        self.assertEqual(session.z.shape, (4,))
        self.assertFalse(session.z.requires_grad)
        snapshot = observation.stored.clone()
        session.remember("another source")
        self.assertTrue(torch.equal(snapshot, observation.stored))
        self.assertTrue(torch.isfinite(session.next_token_logits("target")).all())
        session.reset()
        self.assertEqual(session.z.abs().sum().item(), 0)

    def test_context_and_layer_validation(self):
        with self.assertRaises(ValueError):
            self.transformer.trace("source", [3])
        with self.assertRaises(ValueError):
            self.transformer.next_token_logits("source", 0, lambda h: h)
        with self.assertRaisesRegex(ValueError, "context window"):
            LatentRelay(self.transformer, 1, 1).generate("source", 32)


if __name__ == "__main__":
    unittest.main()
