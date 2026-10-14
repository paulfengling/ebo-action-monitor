import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from ebo.common import config, write_rows
from ebo.data import validate, split_dataset, verify_splits
from ebo.rules import StableAction, SOP
from ebo.video import Packet, Stream
from ebo.training import metrics


def clip(path, color=20, frames=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("test video encoder missing")
    for i in range(frames):
        frame = np.full((48, 64, 3), color, np.uint8)
        cv2.circle(frame, (10+i, 20), 5, (30, 210, 90), -1)
        writer.write(frame)
    writer.release()


class RulesTest(unittest.TestCase):
    def test_stability_and_unknown_reset(self):
        stable = StableAction(.7, 2)
        self.assertIsNone(stable.update("operate", .9))
        self.assertEqual(stable.update("operate", .9), "operate")
        self.assertIsNone(stable.update("operate", .4))
        self.assertIsNone(stable.update("operate", .9))

    def test_sop_unknown_time_not_counted_and_wrong_order(self):
        cfg = config()
        cfg["sop"]["enabled"] = True
        cfg["sop"]["steps"] = [{"action": "operate", "min_seconds": 2}]
        rule = SOP(cfg)
        rule.update("operate", 0)
        rule.update("operate", .5)
        rule.update(None, 1)
        rule.update("operate", 10)
        self.assertEqual(rule.durations, [.5])
        self.assertEqual(rule.update("wrong_operation", 11)["status"], "failed")

    def test_sop_correct_sequence_completes(self):
        cfg = config()
        cfg["sop"]["enabled"] = True
        cfg["sop"]["steps"] = [{"action": "pickup_tool", "min_seconds": .5}, {"action": "operate", "min_seconds": .5}]
        rule = SOP(cfg)
        rule.update("pickup_tool", 0)
        rule.update("pickup_tool", .5)
        self.assertEqual(rule.update("pickup_tool", .75)["alerts"], [])
        rule.update("operate", 1)
        self.assertEqual(rule.update("operate", 1.5)["status"], "completed")

    def test_confusion_metrics(self):
        result = metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])


class DataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = config()
        clip(self.root / "a.mp4")
        self.row = {"video": "a.mp4", "label": "idle", "group_id": "p1", "reviewed": True}

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, name, row):
        path = self.root / name
        write_rows(path, [row])
        return path

    def test_unreviewed_rejected(self):
        path = self.manifest("m.jsonl", {**self.row, "reviewed": False})
        with self.assertRaises(ValueError):
            validate(path, self.cfg)

    def test_synthetic_rejected(self):
        path = self.manifest("m.jsonl", {**self.row, "synthetic": True})
        with self.assertRaises(ValueError):
            validate(path, self.cfg)

    def test_unknown_label_rejected(self):
        path = self.manifest("m.jsonl", {**self.row, "label": "unknown"})
        with self.assertRaises(ValueError):
            validate(path, self.cfg)

    def test_same_video_across_split_rejected_even_with_different_groups(self):
        a = self.manifest("train.jsonl", self.row)
        b = self.manifest("test.jsonl", {**self.row, "group_id": "p2"})
        with self.assertRaises(ValueError):
            verify_splits([a, b], self.cfg)

    def test_source_overlap_rejected(self):
        clip(self.root / "b.mp4", color=40)
        a = self.manifest("train.jsonl", {**self.row, "source_video": "origin", "start": 0, "end": 3})
        b = self.manifest("test.jsonl", {**self.row, "video": "b.mp4", "group_id": "p2",
                                               "source_video": "origin", "start": 2, "end": 4})
        with self.assertRaises(ValueError):
            verify_splits([a, b], self.cfg)

    def test_split_by_groups_with_class_coverage(self):
        rows = []
        for group in range(3):
            for i, c in enumerate(self.cfg["classes"]):
                path = self.root / f"{group}_{i}.mp4"
                clip(path, color=30+group*40+i*8)
                rows.append({"video": str(path), "label": c["id"], "group_id": str(group), "reviewed": True})
        write_rows(self.root / "all.jsonl", rows)
        counts = split_dataset(self.root / "all.jsonl", self.root / "splits", self.cfg)
        self.assertEqual(set(counts), {"train", "val", "test"})
        verify_splits([self.root / "splits" / f"{s}.jsonl" for s in ("train", "val", "test")], self.cfg)


class StreamTest(unittest.TestCase):
    def test_windows_require_continuity_and_clear_on_disconnect(self):
        cfg = config()
        stream = Stream("missing", cfg)
        frame = np.zeros((48, 64, 3), np.uint8)
        stream.file = True
        for t in np.arange(0, 2.1, .0625):
            stream.buffer.append(Packet(0, float(t), frame))
        self.assertIsNotNone(stream.window())
        self.assertEqual(len(stream.window()[2]), 16)
        stream._disconnect()
        self.assertIsNone(stream.window())
        self.assertEqual(stream.epoch, 1)

    def test_stalled_rtsp_never_reuses_last_window(self):
        stream = Stream("rtsp://local-test", config())
        frame = np.zeros((48, 64, 3), np.uint8)
        old = time.monotonic() - 5
        for t in np.arange(old-2, old+.01, .0625):
            stream.buffer.append(Packet(0, float(t), frame))
        stream.latest = stream.buffer[-1]
        self.assertIsNone(stream.window())
        self.assertEqual(stream.snapshot()[1], "stalled")

    def test_rtsp_reconnect_retries_without_accumulating_old_frames(self):
        cfg = config()
        cfg["stream"]["reconnect_seconds"] = .01
        cap = unittest.mock.Mock()
        cap.isOpened.return_value = False
        with patch("ebo.video.open_video", return_value=cap) as opener:
            stream = Stream("rtsp://local-test", cfg).start()
            time.sleep(.06)
            stream.close()
            self.assertGreater(opener.call_count, 1)
            self.assertIsNone(stream.window())
            self.assertEqual(stream.status, "disconnected")


if __name__ == "__main__":
    unittest.main()
