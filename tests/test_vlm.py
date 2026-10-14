import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ebo.common import config, write_rows, rows
from ebo.vlm import propose


class Response:
    def __enter__(self):
        import io
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"label":"operate","reason":"工具正在接触目标"}'}}]}).encode())

    def __exit__(self, *args):
        pass


class VLMTest(unittest.TestCase):
    def test_local_proposal_is_always_unreviewed(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/"input.jsonl", Path(folder)/"output.jsonl"
            write_rows(source, [{"video": "demo.mp4", "label": "idle", "group_id": "p1", "reviewed": True}])
            frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 8
            with patch("ebo.vlm.read_window", return_value=frames), patch("urllib.request.urlopen", return_value=Response()):
                propose(source, target, config(), "http://127.0.0.1:8000/v1/chat/completions", "local")
            proposal = rows(target)[0]
            self.assertIs(proposal["reviewed"], False)
            self.assertEqual(proposal["proposed_label"], "operate")
            self.assertEqual(proposal["label"], "idle")
