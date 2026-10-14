import time
from pathlib import Path
import cv2
from .common import write_rows
from .video import Stream


def record(source, cfg, output, seconds=60, clip_seconds=10, fps=15):
    if seconds <= 0 or clip_seconds <= 0 or fps <= 0:
        raise ValueError("录制时长和帧率必须大于零")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("record_*.mp4")):
        raise ValueError("录制目录已有录像，请使用新目录")
    recording_cfg = {**cfg, "stream": {**cfg["stream"], "buffer_long_edge": 0}}
    reader = Stream(source, recording_cfg, keep_window=False).start()
    start, next_tick = time.monotonic(), time.monotonic()
    writer, writer_epoch, writer_start, writer_count = None, None, None, 0
    records = []
    current_path = None

    def close_writer():
        nonlocal writer
        if writer:
            writer.release()
            writer = None
            records.append({"video": str(current_path.resolve()), "frames": writer_count,
                            "fps": fps, "duration": writer_count / fps,
                            "timing": "receive-clock-resampled"})

    try:
        while time.monotonic() - start < seconds:
            packet, status, epoch = reader.snapshot()
            if packet is None or status != "connected":
                close_writer()
                if reader.ended:
                    break
                time.sleep(.03)
                continue
            now = time.monotonic()
            if writer is not None and (epoch != writer_epoch or now - writer_start >= clip_seconds):
                close_writer()
            if now < next_tick:
                time.sleep(min(.03, next_tick-now))
                continue
            # Do not write stale images while waiting for an RTSP read timeout.
            if not reader.file and now - packet.timestamp > cfg["stream"]["max_gap_seconds"]:
                close_writer()
                time.sleep(.03)
                continue
            if writer is None:
                current_path = output / f"record_{len(records):05d}.mp4"
                h, w = packet.frame.shape[:2]
                writer = cv2.VideoWriter(str(current_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise ValueError("录像编码器不可用")
                writer_epoch, writer_start, writer_count = epoch, now, 0
            writer.write(packet.frame)
            writer_count += 1
            next_tick = now + 1 / fps
            if reader.ended:
                break
    finally:
        close_writer()
        reader.close()
    write_rows(output / "recordings.jsonl", records)
    return {"files": len(records), "frames": sum(r["frames"] for r in records)}
