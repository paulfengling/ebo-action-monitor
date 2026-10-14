"""Causal, exhaustive local-video inference and auditable per-frame exports.

Unlike the live ``Stream`` consumer, this path never skips decoding source frames.
The model sees only a trailing window. Human reference labels are an export-only
annotation; they are deliberately applied after inference and SOP evaluation.
"""
import copy
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from .common import save_json
from .model import device, load_checkpoint, tensor, transform
from .rules import SOP, StableAction


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_rows(path, source):
    """Read half-open intervals; no labels are passed to a model or rule engine."""
    if path is None:
        return []
    intervals = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            video_name = row.get("video_name") or row.get("video")
            if video_name and Path(video_name).name != source.name:
                continue
            try:
                start = float(row.get("start_sec", row.get("start", "")))
                end = float(row.get("end_sec", row.get("end", "")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"参考时间表第 {line} 行缺少有效 start_sec/end_sec 或 start/end") from exc
            label = row.get("step_id") or row.get("label") or row.get("action_label")
            if not label or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"参考时间表第 {line} 行的区间或标签无效")
            intervals.append((start, end, label))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise ValueError("同一视频的参考区间重叠，无法导出唯一 reference_label")
    return intervals


def _font(size, explicit=None):
    # Chinese text needs a CJK font; never silently replace it with tofu glyphs.
    from PIL import ImageFont
    candidates = [explicit] if explicit else [
        os.environ.get("EBO_OVERLAY_FONT"),
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                return ImageFont.truetype(str(candidate), size), str(candidate)
            except OSError:
                continue
    raise ValueError("找不到中文叠加字体；请通过 --overlay-font 或 EBO_OVERLAY_FONT 指定 CJK 字体，"
                     "或关闭叠加视频与逐帧图片导出")


class _Overlay:
    def __init__(self, width, font_path, warning):
        self.font, self.font_path = _font(max(13, min(28, width // 58)), font_path)
        self.warning = warning

    def draw(self, frame, record):
        from PIL import Image, ImageDraw
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image, "RGBA")
        title = (f"EBo 逐帧推理 | 帧 {record['frame_index']} | "
                 f"视频时间 {record['timestamp_seconds']:.3f}s")
        if record["warmup"]:
            state = "预热：历史视频不足一个完整窗口；无模型预测"
            lines = [title, state]
        else:
            confidence = record["confidence"]
            lines = [title, f"稳定动作：{record['action_name']} | 候选：{record['candidate_name']} "
                     f"({confidence:.1%})",
                     f"预测来源帧：{record['prediction_frame_index']} | "
                     f"{'复用上次预测' if record['prediction_reused'] else '本帧执行模型'} | "
                     f"SOP：{record['sop']['status']}"]
        if record.get("reference_label") is not None:
            lines.append(f"人工参考（非预测）：{record['reference_label']}")
        if self.warning:
            lines.append(self.warning)
        gap = self.font.size + 10
        panel_height = min(image.height, 12 + gap * len(lines))
        draw.rectangle((0, 0, image.width, panel_height), fill=(12, 18, 28, 215))
        for i, line in enumerate(lines):
            color = (255, 190, 75, 255) if self.warning and i == len(lines) - 1 else (245, 248, 252, 255)
            draw.text((10, 7 + i * gap), line, font=self.font, fill=color)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _csv_row(record, class_ids):
    result = {key: value for key, value in record.items()
              if key not in ("probabilities", "sop", "sampled_frame_indices")}
    result["sampled_frame_indices"] = json.dumps(record["sampled_frame_indices"])
    result["sop"] = json.dumps(record["sop"], ensure_ascii=False)
    result.update({f"probability_{label}": record["probabilities"][label] for label in class_ids})
    return result


def run(checkpoint, source, out_dir, device_name="auto", stride=1, save_frames=False,
        annotated=True, reference_timeline=None, allow_demo=False, overlay_font=None,
        on_progress=None, max_frames=None, save_raw_frames=False):
    """Infer once per source frame (default) and save CSV, JSONL and summary JSON.

    ``stride > 1`` runs the model only every N frames after warmup; intervening
    rows explicitly reuse its last result, without advancing stability or SOP.
    Stability confirmations use checkpoint ``stream.inference_interval`` in
    source-video time, not the number of full-rate frame predictions. A changed
    candidate or low confidence invalidates the stable action immediately.
    ``save_frames`` writes annotated JPEGs; ``save_raw_frames`` writes untouched
    decoded JPEGs separately. ``annotated`` writes ``annotated.mp4``. Output must
    not already exist, including an empty directory. The returned dict is also
    saved as ``summary.json``. ``on_progress`` receives each exported frame row.

    Time is frame_index / container FPS, a constant-frame-rate timeline rather
    than decoded PTS. Variable-frame-rate inputs must be converted to CFR first
    if exact media timestamps are required. Reference CSV intervals are [start,
    end); accepted columns: start_sec/end_sec/step_id or start/end/label, and
    optionally video_name/video for multi-video tables.
    """
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride 必须为正整数；1 表示每帧均执行模型")
    if max_frames is not None and (not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1):
        raise ValueError("max_frames 必须为正整数")
    source, checkpoint, output = Path(source).resolve(), Path(checkpoint).resolve(), Path(out_dir).resolve()
    if not source.is_file() or not checkpoint.is_file():
        raise ValueError("离线推理需要存在的本地视频文件和 checkpoint 文件")
    if output.exists():
        raise FileExistsError(f"为防止覆盖，请使用新的输出目录：{output}")
    references = _reference_rows(reference_timeline, source)
    target_device = device(device_name)
    model, saved = load_checkpoint(checkpoint, target_device, allow_demo)
    cfg = copy.deepcopy(saved["config"])
    model_cfg, stream_cfg = cfg["model"], cfg["stream"]
    classes = cfg["classes"]
    ids = [entry["id"] for entry in classes]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("checkpoint 的动作类别为空或重复")
    duration, frame_samples = float(model_cfg["window_seconds"]), int(model_cfg["frames"])
    if duration <= 0 or frame_samples < 1:
        raise ValueError("checkpoint 的时间窗口或采样帧数无效")
    processor = transform(model_cfg)
    stable = StableAction(stream_cfg["confidence_threshold"], stream_cfg["stable_predictions"])
    sop = SOP(cfg)
    scope = saved.get("evidence_scope", "unspecified")
    calibration_only = bool(saved.get("calibration_only", False))
    warning = ("同视频训练回放，仅用于流程调通，不代表独立准确率"
               if scope == "same_video_replay" else
               "校准模型回放，不代表独立验证结果" if calibration_only else
               "合成数据演示模型，结果不可用于现场判定" if saved.get("demo_only") else "")
    cap = cv2.VideoCapture(str(source))
    writer = None
    started = time.monotonic()
    summary = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("视频无法打开，或 FPS/尺寸元数据无效")
        overlay = _Overlay(width, overlay_font, warning) if annotated or save_frames else None
        output.mkdir(parents=True, exist_ok=False)
        if save_frames:
            (output / "frames").mkdir()
        if save_raw_frames:
            (output / "raw_frames").mkdir()
        provenance_keys = ("schema_version", "epoch", "pretrained_origin", "manifest_hashes",
                           "training_provenance", "calibration_provenance", "source_video_sha256")
        summary = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source": {"path": str(source), "sha256": _sha256(source), "bytes": source.stat().st_size,
                       "fps": fps, "width": width, "height": height, "declared_frames": expected_frames},
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint),
                           "provenance": {key: saved[key] for key in provenance_keys if key in saved}},
            "device": str(target_device), "evidence_scope": scope, "calibration_only": calibration_only,
            "demo_only": bool(saved.get("demo_only", False)), "warning": warning,
            "independent_accuracy_evaluated": False,
            "class_order": ids, "classes": classes, "checkpoint_config": cfg,
            "stride": stride, "frames_decoded": 0, "frames_exported": 0, "predictions": 0,
            "warmup_frames": 0, "reused_prediction_frames": 0,
            "annotated_frames_written": 0, "annotated_jpegs_saved": 0, "raw_jpegs_saved": 0,
            "timestamp_basis": "frame_index / container_fps (CFR; not decoded presentation timestamps)",
            "sampling": "trailing [timestamp-window_seconds, timestamp), nearest source frame; no future frames",
            "stability_and_sop_updates": "new model predictions only; stability confirmation cadence uses "
                "stream.inference_interval in source time, changed/low-confidence candidates reset immediately; "
                "reused rows do not advance stability or SOP",
            "reference_timeline": ({"path": str(Path(reference_timeline).resolve()),
                                    "sha256": _sha256(reference_timeline),
                                    "purpose": "export-only human reference; never a model input or SOP trigger",
                                    "interval_semantics": "[start,end)"} if reference_timeline else None),
            "outputs": {"csv": "predictions.csv", "jsonl": "predictions.jsonl",
                        "annotated_video": "annotated.mp4" if annotated else None,
                        "frames": "frames/" if save_frames else None,
                        "raw_frames": "raw_frames/" if save_raw_frames else None},
            "overlay_font": overlay.font_path if overlay else None,
        }
        save_json(output / "summary.json", summary)
        if annotated:
            # Pad odd source dimensions because common MP4 codecs require even sizes.
            encoded_size = (width + width % 2, height + height % 2)
            writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, encoded_size)
            if not writer.isOpened():
                raise RuntimeError("无法创建 annotated.mp4，请检查本机 OpenCV 的 MP4 编码支持")
            summary["annotated_encoded_size"] = list(encoded_size)
        history = deque(maxlen=math.ceil(duration * fps) + 3)
        first_prediction_frame = math.ceil(duration * fps - 1e-9)
        last_prediction, last_sop = None, sop.update(None, 0)
        next_confirmation_time = None
        action_counts, candidate_counts = Counter(), Counter()
        total_latency = 0.0
        frame_index = 0
        with open(output / "predictions.csv", "x", encoding="utf-8-sig", newline="") as csv_handle, \
                open(output / "predictions.jsonl", "x", encoding="utf-8") as jsonl_handle:
            csv_writer = None
            while max_frames is None or frame_index < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                summary["frames_decoded"] += 1
                if frame.shape[:2] != (height, width):
                    raise ValueError("本地视频中途改变分辨率，无法保存完整一致的叠加视频")
                timestamp = frame_index / fps
                history.append((frame_index, frame))
                warmup = frame_index < first_prediction_frame
                infer_now = not warmup and (last_prediction is None or
                            (frame_index - first_prediction_frame) % stride == 0)
                if infer_now:
                    targets = (timestamp - duration + np.arange(frame_samples) * duration / frame_samples) * fps
                    indices = np.rint(targets).astype(int).tolist()
                    # Sampling uses the original decoded frames, not overlay pixels.
                    by_index = dict(history)
                    if min(indices) < 0 or max(indices) > frame_index or any(i not in by_index for i in indices):
                        raise RuntimeError("离线因果窗口采样失败：需要的帧不在已解码历史中")
                    frames = [by_index[index] for index in indices]
                    before = time.monotonic()
                    with torch.inference_mode():
                        probabilities = model(tensor(frames, processor, model_cfg.get("roi"))
                                              .unsqueeze(0).to(target_device)).softmax(1)[0].cpu()
                    latency = time.monotonic() - before
                    if len(probabilities) != len(ids) or not torch.isfinite(probabilities).all():
                        raise ValueError("模型输出类别数量不符或包含非有限概率")
                    candidate_index = int(probabilities.argmax())
                    candidate, confidence = classes[candidate_index], float(probabilities[candidate_index])
                    stability_updated = False
                    if confidence < stable.threshold:
                        stable.reset()
                        next_confirmation_time = None
                        action = None
                        stability_updated = True
                    elif candidate["id"] != stable.candidate:
                        action = stable.update(candidate["id"], confidence)
                        next_confirmation_time = timestamp + stream_cfg["inference_interval"]
                        stability_updated = True
                    elif next_confirmation_time is None or timestamp + 1e-9 >= next_confirmation_time:
                        action = stable.update(candidate["id"], confidence)
                        next_confirmation_time = timestamp + stream_cfg["inference_interval"]
                        stability_updated = True
                    else:
                        action = stable.candidate if stable.count >= stable.confirmations else None
                    last_sop = sop.update(action, timestamp)
                    last_prediction = {
                        "candidate": candidate["id"], "candidate_name": candidate["name"],
                        "action": action or "unknown",
                        "action_name": candidate["name"] if action else "无法确认",
                        "confidence": confidence,
                        "decision": candidate["kind"] if action else "unknown",
                        "probabilities": {label: float(probabilities[i]) for i, label in enumerate(ids)},
                        "prediction_frame_index": frame_index,
                        "window_start_frame_index": indices[0], "window_end_frame_index": indices[-1],
                        "window_start_seconds": indices[0] / fps, "window_end_seconds": indices[-1] / fps,
                        "window_requested_start_seconds": timestamp - duration,
                        "window_requested_end_seconds": timestamp,
                        "sampled_frame_indices": indices, "latency_seconds": latency,
                        "stable_confirmation_count": stable.count,
                        "stability_updated": stability_updated,
                    }
                    summary["predictions"] += 1
                    total_latency += latency
                if warmup:
                    prediction = {"candidate": "unknown", "candidate_name": "无法确认",
                                  "action": "unknown", "action_name": "无法确认", "confidence": None,
                                  "decision": "unknown", "probabilities": {label: None for label in ids},
                                  "prediction_frame_index": None, "sampled_frame_indices": [],
                                  "window_start_frame_index": None, "window_end_frame_index": None,
                                  "window_start_seconds": None, "window_end_seconds": None,
                                  "window_requested_start_seconds": None, "window_requested_end_seconds": None,
                                  "latency_seconds": None, "stable_confirmation_count": 0,
                                  "stability_updated": False}
                    summary["warmup_frames"] += 1
                else:
                    prediction = dict(last_prediction)
                    if not infer_now:
                        summary["reused_prediction_frames"] += 1
                        prediction["stability_updated"] = False
                record = {"frame_index": frame_index, "timestamp_seconds": timestamp,
                          **prediction, "warmup": warmup,
                          "prediction_reused": bool(not warmup and not infer_now),
                          "prediction_age_seconds": (None if warmup else
                              (frame_index - prediction["prediction_frame_index"]) / fps),
                          "sop": copy.deepcopy(last_sop),
                          "evidence_scope": scope, "calibration_only": calibration_only,
                          "demo_only": bool(saved.get("demo_only", False))}
                # Keep references downstream of *all* model and SOP computation.
                record["reference_label"] = next((label for start, end, label in references
                                                  if start <= timestamp < end), None)
                row = _csv_row(record, ids)
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_handle, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                jsonl_handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                summary["frames_exported"] += 1
                if overlay:
                    rendered = overlay.draw(frame, record)
                    if save_frames:
                        if not cv2.imwrite(str(output / "frames" / f"frame_{frame_index:06d}.jpg"), rendered):
                            raise RuntimeError(f"无法保存帧 {frame_index} 的预测图片")
                        summary["annotated_jpegs_saved"] += 1
                    if writer is not None:
                        if encoded_size != (width, height):
                            rendered = cv2.copyMakeBorder(rendered, 0, height % 2, 0, width % 2,
                                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        writer.write(rendered)
                        summary["annotated_frames_written"] += 1
                if save_raw_frames:
                    if not cv2.imwrite(str(output / "raw_frames" / f"frame_{frame_index:06d}.jpg"), frame):
                        raise RuntimeError(f"无法保存帧 {frame_index} 的原始图片")
                    summary["raw_jpegs_saved"] += 1
                action_counts[record["action"]] += 1
                candidate_counts[record["candidate"]] += 1
                if on_progress:
                    on_progress(copy.deepcopy(record))
                frame_index += 1
        if summary["frames_decoded"] == 0:
            raise ValueError("视频未成功解码任何画面")
        limited = max_frames is not None and frame_index >= max_frames and (
            expected_frames <= 0 or frame_index < expected_frames)
        decode_incomplete = not limited and expected_frames > 0 and frame_index < expected_frames
        summary.update(status="partial" if limited else "decode_incomplete" if decode_incomplete else "completed",
                       frame_limit_reached=limited, decode_incomplete=decode_incomplete,
                       action_frame_counts=dict(action_counts), candidate_frame_counts=dict(candidate_counts),
                       final_sop=last_sop,
                       mean_inference_seconds=(total_latency / summary["predictions"] if summary["predictions"] else None),
                       elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        if summary is not None:
            summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if summary is not None:
            save_json(output / "summary.json", summary)
    return summary
