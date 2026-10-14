import csv
import json
import threading
import time
import uuid
from pathlib import Path
import cv2
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QComboBox, QFileDialog, QListWidget,
                            QDoubleSpinBox, QLineEdit, QMessageBox)
from .common import rows, write_rows
from .data import prepare, resolve_video
from .video import video_info
from .inference import run


def display(label, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    label.setPixmap(QPixmap.fromImage(image).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation))


def push(text, fn):
    button = QPushButton(text)
    button.clicked.connect(fn)
    return button


class Annotation(QMainWindow):
    def __init__(self, cfg, output, manifest=None):
        super().__init__()
        self.cfg, self.output = cfg, Path(output).resolve()
        self.manifest = manifest
        self.items = rows(manifest) if manifest else []
        self.saved = rows(self.output) if self.output.exists() else []
        self.cap = None
        self.current = None
        self.current_item = {}
        self.setWindowTitle("EBo · 视频片段标注与人工确认")
        self.resize(1100, 800)
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("EBo · 训练视频片段标注"))
        layout.addWidget(QLabel("每个片段只包含一个主要动作；确认后用于训练。group_id 建议用人员ID，防止同一人员跨训练/测试。"))
        body = QHBoxLayout()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.select)
        body.addWidget(self.list, 1)
        self.video = QLabel("打开视频文件，或载入待确认标签")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 420)
        self.video.setStyleSheet("background:#162237;color:white;")
        body.addWidget(self.video, 3)
        layout.addLayout(body, 1)
        controls = QHBoxLayout()
        controls.addWidget(push("添加视频", self.add))
        controls.addWidget(push("播放/暂停", self.play))
        self.position = QLabel("0.0 秒")
        controls.addWidget(self.position)
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for spin in (self.start, self.end):
            spin.setDecimals(2)
            spin.setRange(0, 86400)
        controls.addWidget(QLabel("片段开始"))
        controls.addWidget(self.start)
        controls.addWidget(push("用当前时间", lambda: self.start.setValue(self.time())))
        controls.addWidget(QLabel("结束"))
        controls.addWidget(self.end)
        controls.addWidget(push("用当前时间", lambda: self.end.setValue(self.time())))
        layout.addLayout(controls)
        form = QHBoxLayout()
        self.labels = QComboBox()
        for c in cfg["classes"]:
            self.labels.addItem(c["name"] + " / " + c["id"], c["id"])
        form.addWidget(self.labels)
        self.group = QLineEdit()
        self.group.setPlaceholderText("group_id：人员ID或独立采集分组")
        form.addWidget(self.group)
        form.addWidget(push("切片并人工确认标签", self.save))
        layout.addLayout(form)
        self.note = QLabel("不会自动接受大模型标签。")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.reload_list()

    def reload_list(self):
        self.list.clear()
        for row in self.items:
            self.list.addItem(Path(row["video"]).name + " · " + str(row.get("proposed_label", row.get("label", "未标注"))))

    def add(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "添加视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        self.items.extend({"video": str(Path(p).resolve())} for p in paths)
        self.reload_list()

    def select(self, index):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        if index < 0:
            return
        item = self.items[index]
        self.current_item = item
        path = resolve_video(item, self.manifest) if self.manifest else Path(item["video"])
        self.current = str(path)
        try:
            info = video_info(path)
            self.cap = cv2.VideoCapture(str(path))
            self.start.setValue(0)
            self.end.setValue(info["seconds"])
            self.group.setText(str(item.get("group_id", "")))
            proposed = item.get("proposed_label", item.get("label"))
            selected = self.labels.findData(proposed)
            if selected >= 0:
                self.labels.setCurrentIndex(selected)
            self.note.setText(("合成验证视频 · 不用于现场训练。 " if item.get("synthetic") else "") +
                             "候选标签（需人工确认）：" + str(proposed) + "；" + str(item.get("proposal_reason", "")))
            self.tick()
        except Exception as exc:
            self.note.setText("视频打开失败：" + type(exc).__name__)

    def time(self):
        if not self.cap:
            return 0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) / max(1, fps)

    def play(self):
        if not self.cap:
            return
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(max(10, int(1000 / max(1, self.cap.get(cv2.CAP_PROP_FPS)))))

    def tick(self):
        if self.cap:
            ok, frame = self.cap.read()
            if ok:
                display(self.video, frame)
                self.position.setText(f"{self.time():.2f} 秒")
            else:
                self.timer.stop()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def save(self):
        if not self.current or not self.group.text().strip():
            self.note.setText("请先选择视频并填写 group_id。")
            return
        if self.end.value() - self.start.value() < self.cfg["model"]["window_seconds"]:
            self.note.setText("片段时长不能小于模型时间窗口。")
            return
        self.timer.stop()
        folder = self.output.parent / "clips" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        csv_path, json_path = folder / "segment.csv", folder / "clip.jsonl"
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video", "start", "end", "label", "group_id"])
            writer.writeheader()
            writer.writerow({"video": self.current, "start": self.start.value(), "end": self.end.value(),
                             "label": self.labels.currentData(), "group_id": self.group.text().strip()})
        try:
            prepare(csv_path, folder, json_path, reviewed=True)
            new_rows = rows(json_path)
            for row in new_rows:
                row["synthetic"] = bool(self.current_item.get("synthetic", False))
                if "source_video" in self.current_item:
                    origin = float(self.current_item.get("start", 0))
                    row.update(source_video=self.current_item["source_video"],
                               start=origin+self.start.value(), end=origin+self.end.value())
            self.saved.extend(new_rows)
            write_rows(self.output, self.saved)
            self.note.setText(f"已保存 {len(self.saved)} 条人工确认片段。")
        except Exception as exc:
            self.note.setText("保存失败：" + str(exc))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


class InferenceWorker(QThread):
    frame = pyqtSignal(object)
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, options):
        super().__init__()
        self.options, self.stop = options, threading.Event()

    def run(self):
        try:
            run(**self.options, on_frame=self.frame.emit, on_result=self.result.emit, stop_event=self.stop)
        except Exception as exc:
            # Never expose raw source or RTSP credentials in UI errors.
            self.error.emit("推理停止：" + type(exc).__name__ + "；请检查模型、配置与连接")


class Monitor(QMainWindow):
    def __init__(self, options):
        super().__init__()
        self.setWindowTitle("EBo · RTSP 实时动作识别")
        self.resize(1100, 780)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.addWidget(QLabel("EBo · RTSP 视频动作监测"))
        self.video = QLabel("正在连接视频流")
        self.video.setMinimumSize(640, 420)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background:#162237;color:white;")
        layout.addWidget(self.video, 1)
        self.info = QLabel("模型加载中；未知结果不会自动判为正确。")
        self.info.setStyleSheet("font-size:22px;padding:15px;")
        layout.addWidget(self.info)
        self.details = QLabel("等待连续视频窗口")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        layout.addWidget(push("停止", self.close))
        self.alert_dialog = None
        self.last_alert, self.last_alert_time = None, 0
        self.worker = InferenceWorker(options)
        self.worker.frame.connect(lambda frame: display(self.video, frame))
        self.worker.result.connect(self.result)
        self.worker.error.connect(self.info.setText)
        self.worker.start()

    def result(self, obj):
        if obj["type"] == "stream":
            self.info.setText("视频状态：" + obj["status"])
            self.video.clear()
            self.video.setText("等待新视频画面")
            return
        self.info.setText(obj.get("name", "无法确认") + " · " + obj.get("decision", "unknown"))
        sop = obj.get("sop", {})
        self.details.setText(f"候选：{obj.get('candidate', '—')}；置信度：{obj.get('confidence', 0):.2f}；"
                             f"模型耗时：{obj.get('latency_seconds', 0):.3f}s；流程：{sop.get('status', '—')}\n"
                             + ("合成验证模型，不可用于现场判断" if obj.get("demo_only") else obj.get("reason", "")))
        alerts = sop.get("alerts", [])
        if alerts and (alerts != self.last_alert or time.monotonic() - self.last_alert_time > 10):
            if self.alert_dialog:
                self.alert_dialog.close()
            self.alert_dialog = QMessageBox(QMessageBox.Icon.Warning, "动作检查", "；".join(alerts), parent=self)
            self.alert_dialog.setModal(False)
            self.alert_dialog.show()
            self.last_alert, self.last_alert_time = alerts, time.monotonic()

    def closeEvent(self, event):
        self.worker.stop.set()
        if self.worker.isRunning():
            self.info.setText("正在停止视频线程，请稍候……")
            QTimer.singleShot(200, self.close)
            event.ignore()
            return
        event.accept()


def annotate(cfg, output, manifest=None):
    app = QApplication.instance() or QApplication([])
    window = Annotation(cfg, output, manifest)
    window.show()
    return app.exec()


def monitor(options):
    app = QApplication.instance() or QApplication([])
    window = Monitor(options)
    window.show()
    return app.exec()
