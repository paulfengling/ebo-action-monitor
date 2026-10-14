"""Capture runs independently of inference; sampled ring buffer never forms a work queue."""
import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


@dataclass
class Packet:
    epoch: int
    timestamp: float
    frame: object


def open_video(source, stream):
    network = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://"))
    if network:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(stream["open_timeout_ms"]),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(stream["read_timeout_ms"])])
    return cv2.VideoCapture(source)


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or n <= 0:
            raise ValueError("视频无法读取或没有有效帧率")
        return {"fps": fps, "frames": n, "seconds": n / fps}
    finally:
        cap.release()


def read_window(path, model, start=None):
    info = video_info(path)
    duration = model["window_seconds"]
    if start is None:
        start = max(0, (info["seconds"] - duration) / 2)
    if start < 0:
        raise ValueError("起始时间不能小于零")
    indices = np.rint((start + np.arange(model["frames"]) * duration / model["frames"]) * info["fps"]).astype(int)
    indices = np.minimum(indices, info["frames"] - 1)
    cap = cv2.VideoCapture(str(path))
    result = []
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise ValueError("视频窗口解码失败")
            result.append(frame)
    finally:
        cap.release()
    return result


class Stream:
    def __init__(self, source, cfg, keep_window=True):
        self.source, self.cfg = source, cfg
        self.keep_window = keep_window
        self.stop = threading.Event()
        self.lock = threading.Lock()
        sample_fps = cfg["model"]["frames"] / cfg["model"]["window_seconds"] * 2
        self.sample_interval = 1 / sample_fps
        self.buffer = deque(maxlen=int(sample_fps * cfg["model"]["window_seconds"] * 2) + 8)
        self.latest = None
        self.epoch = 0
        self.status = "connecting"
        self.ended = False
        self.file = isinstance(source, str) and Path(source).is_file()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _disconnect(self):
        with self.lock:
            self.buffer.clear()
            self.latest = None
            self.epoch += 1
            self.status = "disconnected"

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    cap = open_video(self.source, self.cfg["stream"])
                except cv2.error:
                    self._disconnect()
                    if self.file:
                        break
                    self.stop.wait(self.cfg["stream"]["reconnect_seconds"])
                    continue
                if not cap.isOpened():
                    cap.release()
                    self._disconnect()
                    if self.file:
                        break
                    self.stop.wait(self.cfg["stream"]["reconnect_seconds"])
                    continue
                last_sample, last_receive, last_shape = -1e9, None, None
                fps = cap.get(cv2.CAP_PROP_FPS)
                if self.file and fps <= 0:
                    cap.release()
                    self._disconnect()
                    break
                file_index, origin = 0, time.monotonic()
                try:
                    while not self.stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if self.file:
                            self.stop.wait(max(0, origin + file_index / fps - time.monotonic()))
                            timestamp = file_index / fps
                            file_index += 1
                        else:
                            timestamp = time.monotonic()
                        if last_receive is not None and timestamp - last_receive > self.cfg["stream"]["max_gap_seconds"]:
                            self._disconnect()
                        last_receive = timestamp
                        if last_shape is not None and frame.shape != last_shape:
                            self._disconnect()
                        last_shape = frame.shape
                        h, w = frame.shape[:2]
                        edge = self.cfg["stream"]["buffer_long_edge"]
                        scale = min(1, edge / max(h, w)) if edge > 0 else 1
                        if scale < 1:
                            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        with self.lock:
                            self.latest = Packet(self.epoch, timestamp, frame)
                            self.status = "connected"
                            if self.keep_window and timestamp - last_sample >= self.sample_interval * 0.9:
                                self.buffer.append(self.latest)
                                last_sample = timestamp
                finally:
                    cap.release()
                if self.file:
                    # Preserve the final valid window until the consumer drains it.
                    break
                self._disconnect()
                self.stop.wait(self.cfg["stream"]["reconnect_seconds"])
        finally:
            self.ended = True

    def snapshot(self):
        with self.lock:
            status = self.status
            if not self.file and self.latest is not None and time.monotonic() - self.latest.timestamp > self.cfg["stream"]["max_gap_seconds"]:
                status = "stalled"
            return self.latest, status, self.epoch

    def window(self):
        with self.lock:
            packets = list(self.buffer)
        if not packets:
            return None
        if not self.file and time.monotonic() - packets[-1].timestamp > self.cfg["stream"]["max_gap_seconds"]:
            return None
        m = self.cfg["model"]
        end = packets[-1].timestamp
        start = end - m["window_seconds"]
        tolerance = self.sample_interval * 1.5
        if packets[0].timestamp > start + tolerance:
            return None
        stamps = np.array([p.timestamp for p in packets])
        if np.any(np.diff(stamps[stamps >= start - tolerance]) > self.cfg["stream"]["max_gap_seconds"]):
            return None
        targets = start + np.arange(m["frames"]) * m["window_seconds"] / m["frames"]
        indices = [int(np.argmin(np.abs(stamps - t))) for t in targets]
        if any(abs(stamps[i] - target) > tolerance for i, target in zip(indices, targets)):
            return None
        return packets[-1].epoch, end, [packets[i].frame for i in indices]

    def close(self):
        self.stop.set()
        self.thread.join((self.cfg["stream"]["open_timeout_ms"] + self.cfg["stream"]["read_timeout_ms"]) / 1000 + 2)
        if self.thread.is_alive():
            raise RuntimeError("视频后端未按超时退出，请检查摄像头/后端")
