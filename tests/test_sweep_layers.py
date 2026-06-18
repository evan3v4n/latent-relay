import tempfile
import unittest
from pathlib import Path

from experiments.sweep_layers import layer_pair, write_results
from metrics import LogitEffect


class SweepLayerTests(unittest.TestCase):
    def test_parses_layer_pair(self) -> None:
        self.assertEqual(layer_pair("4:8"), (4, 8))

    def test_writes_reusable_csv(self) -> None:
        rows = [(4, 8, 0.5, -1.0, LogitEffect(2.0, 1.0, 0.25))]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep.csv"
            write_results(output, rows)
            contents = output.read_text()
        self.assertIn("read_layer,write_layer,alpha,z,l2,max_abs,kl", contents)
        self.assertIn("4,8,0.5,-1.0,2.0,1.0,0.25", contents)


if __name__ == "__main__":
    unittest.main()
