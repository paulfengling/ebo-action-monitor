import unittest
import numpy as np
import torch
from ebo.common import config
from ebo.model import tensor, transform


class PreprocessTest(unittest.TestCase):
    def test_rgb_channel_order_and_model_shape(self):
        cfg = config()["model"]
        frame = np.zeros((160, 200, 3), np.uint8)
        frame[:, :, 2] = 255
        x = tensor([frame]*16, transform(cfg))
        self.assertEqual(tuple(x.shape), (3, 16, 112, 112))
        self.assertGreater(float(x[0].mean()), float(x[2].mean()))

    def test_roi_applied_before_resize_for_all_frames(self):
        cfg = config()["model"]
        frame = np.zeros((160, 200, 3), np.uint8)
        frame[:, 100:, 2] = 255
        x = tensor([frame]*16, transform(cfg), [0.5, 0, 1, 1])
        self.assertTrue(torch.all(x[0] > 1))
