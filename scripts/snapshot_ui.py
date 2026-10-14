"""Reproducible headless visual QA using synthetic smoke-test video, never real recognition."""
import argparse
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from ebo.ui import Annotation, Monitor
from ebo.common import config

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-dir", default="runs/smoke_final")
parser.add_argument("--out", default="runs/ui_check")
args = parser.parse_args()
torch.set_num_threads(2)
base, out = Path(args.smoke_dir).resolve(), Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
app = QApplication([])
ann = Annotation(config(), out/"reviewed.jsonl", base/"test.jsonl")
ann.show()
ann.list.setCurrentRow(0)
options = dict(checkpoint=str(base/"model/best.pt"), source=str(base/"test_0.mp4"),
               runtime_cfg=config(base/"config.yaml"), device_name="cpu", log=str(out/"inference.jsonl"),
               duration=3, allow_demo=True)
mon = Monitor(options)
mon.show()


def save():
    ann.grab().save(str(out/"annotation.png"))
    mon.grab().save(str(out/"monitor.png"))
    ann.close()
    mon.close()


QTimer.singleShot(4000, save)
QTimer.singleShot(12000, app.quit)
app.exec()
print("标注与监测界面截图已保存；使用合成验证视频")
