import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from ebo.common import config, write_rows, rows
from ebo.ui import Annotation
from tests.test_pipeline import clip


class AnnotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_annotation_saves_reviewed_clip_and_preserves_synthetic_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clip(root/"source.mp4", frames=30)
            write_rows(root/"pending.jsonl", [{"video": "source.mp4", "group_id": "p1",
                                              "synthetic": True, "proposed_label": "operate"}])
            ui = Annotation(config(), root/"reviewed.jsonl", root/"pending.jsonl")
            ui.show()
            self.app.processEvents()
            ui.list.setCurrentRow(0)
            ui.end.setValue(2)
            ui.save()
            result = rows(root/"reviewed.jsonl")[0]
            self.assertTrue(result["reviewed"])
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["label"], "operate")
            self.assertTrue(Path(result["video"]).exists())
            ui.close()
            self.app.processEvents()
