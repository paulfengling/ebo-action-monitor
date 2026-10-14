import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from ebo.calibration import absorb_standardization
from ebo.model import load_checkpoint


class CalibrationTest(unittest.TestCase):
    def test_absorbed_head_has_same_logits(self):
        torch.manual_seed(42)
        head = nn.Linear(20, 6)
        x = torch.randn(30, 20)*3+5
        mean, std = x.mean(0), x.std(0)
        merged = absorb_standardization(head, mean, std)
        torch.testing.assert_close(head((x-mean)/std), merged(x), atol=1e-5, rtol=1e-5)

    def test_calibration_checkpoint_rejected_without_explicit_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"fit.pt"
            torch.save({"schema_version": 1, "calibration_only": True, "demo_only": True}, path)
            with self.assertRaisesRegex(ValueError, "单视频"):
                load_checkpoint(path, torch.device("cpu"))
