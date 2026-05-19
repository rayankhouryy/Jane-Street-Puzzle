"""Tests for neurodecomp.emit.

Build a synthetic AND-of-byte-equality network, run emit_to_file, and check:
  - the emitted file is syntactically valid Python
  - the recovered tokenizer length matches the model's input dim
  - the emitted ``encode`` function works
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from neurodecomp import emit
from tests.test_tail import build_byte_eq_network


def _load_emitted_module(path: Path):
    spec = importlib.util.spec_from_file_location("emitted", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmitTests(unittest.TestCase):
    def test_emit_runs_on_minimal_network(self):
        targets = [0xAB, 0xCD, 0xEF, 0x12]
        net = build_byte_eq_network(targets)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "emitted.py"
            summary = emit.emit_to_file(net, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(summary["bytes_written"], 100)
            # Make sure the emitted file imports without errors.
            mod = _load_emitted_module(out_path)
            # The encoded length should match the network's expected input dim.
            self.assertEqual(mod.TOKENIZER_LENGTH, net[0].in_features)
            # The target digest, if any, should match the synthetic targets.
            # synthetic build_byte_eq_network targets pack as hex...
            expected_hex = "".join(f"{t:02x}" for t in targets)
            self.assertEqual(mod.TARGET_DIGEST_HEX, expected_hex)

    def test_emit_marks_pending_round_function(self):
        net = build_byte_eq_network([1, 2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "emitted.py"
            emit.emit_to_file(net, out_path)
            mod = _load_emitted_module(out_path)
            # round_function must raise NotImplementedError; that's the
            # explicit gap marker.
            with self.assertRaises(NotImplementedError):
                mod.round_function(0, 0, 0, 0, 0, 0, 0, 0)

    def test_emit_tokenizer_works(self):
        net = build_byte_eq_network([7, 8, 9])
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "emitted.py"
            emit.emit_to_file(net, out_path)
            mod = _load_emitted_module(out_path)
            # The synthetic net has no cloudpickle tokenizer, so the emitted
            # encode treats the input as bytes via ord(...).
            result = mod.encode("ab")
            self.assertEqual(len(result), mod.TOKENIZER_LENGTH)
            self.assertEqual(result[0], ord("a"))
            self.assertEqual(result[1], ord("b"))


if __name__ == "__main__":
    unittest.main()
