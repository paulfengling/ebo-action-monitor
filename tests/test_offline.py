"""Offline inference tests use real encoded frames and a deterministic fake model."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from ebo.common import config
from ebo.offline import run


class FakeModel:
    def __init__(self, logits=None):
        self.calls = 0
        self.logits = logits or [[5., 0.]]

    def __call__(self, batch):
        output = self.logits[min(self.calls, len(self.logits) - 1)]
        self.calls += 1
        return torch.tensor([output])


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "example.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
        if not writer.isOpened():
            self.skipTest("OpenCV test MP4 encoder unavailable")
        for index in range(20):
            writer.write(np.full((64, 96, 3), index * 10, np.uint8))
        writer.release()
        self.checkpoint = self.root / "model.pt"
        self.checkpoint.write_bytes(b"test checkpoint whose loader is patched")
        cfg = config()
        cfg["classes"] = [{"id": "S1", "name": "安装底座", "kind": "correct"},
                          {"id": "idle", "name": "未操作", "kind": "neutral"}]
        cfg["model"].update(window_seconds=1., frames=4)
        cfg["stream"].update(confidence_threshold=.7, stable_predictions=2, inference_interval=.5)
        cfg["sop"] = {"enabled": True, "steps": [{"action": "S1", "min_seconds": .3}]}
        self.saved = {"config": cfg, "demo_only": False, "schema_version": 1,
                      "calibration_only": True, "evidence_scope": "same_video_replay"}
        self.observed_windows = []

    def tearDown(self):
        self.temp.cleanup()

    def infer(self, directory, model=None, **kwargs):
        model = model or FakeModel()

        def fake_tensor(frames, processor, roi):
            self.observed_windows.append([int(frame.mean()) for frame in frames])
            return torch.zeros((3, 4, 8, 8))

        with patch("ebo.offline.load_checkpoint", return_value=(model, copy.deepcopy(self.saved))), \
                patch("ebo.offline.transform", return_value=None), \
                patch("ebo.offline.tensor", side_effect=fake_tensor):
            summary = run(self.checkpoint, self.video, self.root / directory, device_name="cpu",
                          annotated=False, **kwargs)
        rows = [json.loads(line) for line in (self.root / directory / "predictions.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        return summary, rows, model

    def test_every_decoded_frame_has_causal_prediction_and_warmup(self):
        summary, rows, model = self.infer("full")
        self.assertEqual(len(rows), 20)
        self.assertEqual([r["frame_index"] for r in rows], list(range(20)))
        self.assertEqual(summary["frames_exported"], 20)
        self.assertEqual(summary["predictions"], 10)
        self.assertEqual(model.calls, 10)
        self.assertEqual(summary["warmup_frames"], 10)
        self.assertEqual(summary["status"], "completed")
        for row in rows[:10]:
            self.assertTrue(row["warmup"])
            self.assertEqual(row["action"], "unknown")
            self.assertIsNone(row["confidence"])
            self.assertIsNone(row["prediction_frame_index"])
            self.assertEqual(row["probabilities"], {"S1": None, "idle": None})
        for row in rows[10:]:
            self.assertEqual(row["prediction_frame_index"], row["frame_index"])
            self.assertFalse(row["prediction_reused"])
            self.assertTrue(all(0 <= i <= row["frame_index"] for i in row["sampled_frame_indices"]))
            self.assertAlmostEqual(sum(row["probabilities"].values()), 1, places=6)
        self.assertEqual(rows[10]["sampled_frame_indices"], [0, 2, 5, 8])
        self.assertEqual(rows[19]["sampled_frame_indices"], [9, 12, 14, 16])
        self.assertAlmostEqual(rows[19]["timestamp_seconds"], 1.9)
        self.assertEqual(rows[14]["action"], "unknown")
        self.assertEqual(rows[15]["action"], "S1")
        self.assertEqual(rows[15]["stable_confirmation_count"], 2)
        self.assertFalse(rows[14]["stability_updated"])
        self.assertIn("不代表独立准确率", summary["warning"])
        self.assertFalse(summary["independent_accuracy_evaluated"])
        self.assertEqual(len(summary["source"]["sha256"]), 64)
        self.assertEqual(len(summary["checkpoint"]["sha256"]), 64)
        with open(self.root / "full" / "predictions.csv", encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 20)
        self.assertIn("probability_S1", csv_rows[0])
        self.assertEqual(csv_rows[0]["confidence"], "")

    def test_stride_reuses_but_does_not_add_stability_confirmations(self):
        summary, rows, model = self.infer("stride", stride=3)
        self.assertEqual(model.calls, 4)
        self.assertEqual(summary["predictions"], 4)
        self.assertEqual(summary["reused_prediction_frames"], 6)
        self.assertEqual([r["frame_index"] for r in rows if not r["warmup"] and not r["prediction_reused"]],
                         [10, 13, 16, 19])
        self.assertEqual(rows[15]["action"], "unknown")
        self.assertEqual(rows[16]["action"], "S1")
        self.assertEqual(rows[15]["stable_confirmation_count"], 1)
        self.assertFalse(rows[15]["stability_updated"])
        self.assertEqual(rows[11]["sop"], rows[10]["sop"])
        self.assertEqual(rows[17]["sop"], rows[16]["sop"])
        self.assertEqual(rows[18]["prediction_frame_index"], 16)
        self.assertAlmostEqual(rows[18]["prediction_age_seconds"], .2)

    def test_changed_candidate_and_low_confidence_reset_immediately(self):
        model = FakeModel([[5., 0.]] * 6 + [[0., 5.], [0., 0.], [5., 0.], [5., 0.]])
        _, rows, _ = self.infer("switch", model=model)
        self.assertEqual(rows[15]["action"], "S1")
        self.assertEqual(rows[16]["candidate"], "idle")
        self.assertEqual(rows[16]["action"], "unknown")
        self.assertEqual(rows[17]["action"], "unknown")
        self.assertEqual(rows[17]["stable_confirmation_count"], 0)
        self.assertEqual(rows[18]["stable_confirmation_count"], 1)
        self.assertEqual(rows[19]["action"], "unknown")

    def test_reference_labels_do_not_change_model_stability_or_sop(self):
        reference = self.root / "reference.csv"
        reference.write_text("video_name,start_sec,end_sec,step_id\n"
                             "example.mp4,0,1.5,DELIBERATELY_WRONG\n"
                             "other.mp4,0,5,IGNORED\n", encoding="utf-8")
        _, original, _ = self.infer("no_reference")
        _, referenced, _ = self.infer("with_reference", reference_timeline=reference)
        self.assertEqual(referenced[0]["reference_label"], "DELIBERATELY_WRONG")
        self.assertIsNone(referenced[15]["reference_label"])
        for before, after in zip(original, referenced):
            for key in ("action", "candidate", "probabilities", "sop", "stable_confirmation_count"):
                self.assertEqual(before[key], after[key])

    def test_existing_output_is_never_overwritten(self):
        (self.root / "already_exists").mkdir()
        with self.assertRaises(FileExistsError):
            run(self.checkpoint, self.video, self.root / "already_exists", annotated=False)

    def test_max_frames_reports_partial_without_fake_model_prediction(self):
        summary, rows, model = self.infer("partial", max_frames=5)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(len(rows), 5)
        self.assertEqual(model.calls, 0)
        self.assertEqual(summary["predictions"], 0)

    def test_frames_and_video_include_every_source_frame(self):
        overlay = Mock()
        overlay.font_path = "test CJK font"
        overlay.draw.side_effect = lambda frame, row: frame.copy()
        with patch("ebo.offline._Overlay", return_value=overlay), \
                patch("ebo.offline.load_checkpoint", return_value=(FakeModel(), copy.deepcopy(self.saved))), \
                patch("ebo.offline.transform", return_value=None), \
                patch("ebo.offline.tensor", return_value=torch.zeros((3, 4, 8, 8))):
            summary = run(self.checkpoint, self.video, self.root / "rendered", device_name="cpu",
                          annotated=True, save_frames=True, save_raw_frames=True)
        self.assertEqual(len(list((self.root / "rendered" / "frames").glob("*.jpg"))), 20)
        self.assertEqual(len(list((self.root / "rendered" / "raw_frames").glob("*.jpg"))), 20)
        cap = cv2.VideoCapture(str(self.root / "rendered" / "annotated.mp4"))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        self.assertEqual(count, 20)
        self.assertEqual(overlay.draw.call_count, 20)
        self.assertEqual(summary["status"], "completed")

    def test_failure_closes_video_and_records_partial_export_count(self):
        video_writer = Mock()
        video_writer.isOpened.return_value = True
        overlay = Mock()
        overlay.font_path = "test CJK font"
        overlay.draw.side_effect = lambda frame, row: frame.copy()
        failed_model = Mock(side_effect=RuntimeError("intentional inference failure"))
        with patch("ebo.offline.cv2.VideoWriter", return_value=video_writer), \
                patch("ebo.offline._Overlay", return_value=overlay), \
                patch("ebo.offline.load_checkpoint", return_value=(failed_model, copy.deepcopy(self.saved))), \
                patch("ebo.offline.transform", return_value=None), \
                patch("ebo.offline.tensor", return_value=torch.zeros((3, 4, 8, 8))):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                run(self.checkpoint, self.video, self.root / "failed", device_name="cpu", annotated=True)
        video_writer.release.assert_called_once()
        summary = json.loads((self.root / "failed" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["frames_exported"], 10)
        self.assertEqual(summary["frames_decoded"], 11)


if __name__ == "__main__":
    unittest.main()
