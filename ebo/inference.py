import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
from .model import device, load_checkpoint, transform, tensor
from .rules import StableAction, SOP
from .video import Stream


def run(checkpoint, source, runtime_cfg, device_name="auto", log=None, duration=None,
        allow_demo=False, on_result=None, on_frame=None, stop_event=None):
    target_device = device(device_name)
    model, saved = load_checkpoint(checkpoint, target_device, allow_demo)
    cfg = saved["config"].copy()
    # Class order / preprocessing stay coupled to checkpoint, never the display config.
    if runtime_cfg["classes"] != cfg["classes"] or runtime_cfg["model"] != cfg["model"]:
        raise ValueError("运行配置的类别/预处理与 checkpoint 不一致，请使用训练时配置")
    cfg["stream"], cfg["sop"] = runtime_cfg["stream"], runtime_cfg["sop"]
    processor = transform(cfg["model"])
    stable = StableAction(cfg["stream"]["confidence_threshold"], cfg["stream"]["stable_predictions"])
    sop = SOP(cfg)
    capture = Stream(source, cfg).start()
    handle = None
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        handle = open(log, "w", encoding="utf-8")
    start, last_processed, epoch, last_frame_time = time.monotonic(), -1e9, -1, None
    last_status, last_wait = None, -1e9
    predictions = 0

    def emit(obj):
        obj.update(wall_time=datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                   demo_only=saved["demo_only"])
        if handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
            handle.flush()
        if on_result:
            on_result(obj)
        else:
            print(json.dumps(obj, ensure_ascii=False), flush=True)

    try:
        while not (stop_event and stop_event.is_set()):
            if duration is not None and time.monotonic() - start >= duration:
                break
            packet, status, current_epoch = capture.snapshot()
            if current_epoch != epoch:
                stable.reset()
                sop.reset()
                last_processed, epoch = -1e9, current_epoch
            if packet is not None and packet.timestamp != last_frame_time:
                if on_frame:
                    on_frame(packet.frame)
                last_frame_time = packet.timestamp
            if status != last_status:
                emit({"type": "stream", "status": status, "action": "unknown", "epoch": epoch})
                last_status = status
            window = capture.window()
            if window is None:
                if time.monotonic() - last_wait >= 1:
                    stable.reset()
                    sop.previous = None
                    emit({"type": "prediction", "action": "unknown", "decision": "unknown",
                          "reason": "等待完整连续窗口", "epoch": epoch})
                    last_wait = time.monotonic()
                if capture.ended:
                    break
                time.sleep(.03)
                continue
            window_epoch, end, frames = window
            if end - last_processed < cfg["stream"]["inference_interval"]:
                if capture.ended:
                    break
                time.sleep(.03)
                continue
            last_processed = end
            inference_start = time.monotonic()
            with torch.inference_mode():
                probabilities = model(tensor(frames, processor, cfg["model"].get("roi")).unsqueeze(0).to(target_device)).softmax(1)[0].cpu()
            latest, _, now_epoch = capture.snapshot()
            latency = time.monotonic() - inference_start
            if now_epoch != window_epoch or latest is None or latest.timestamp - end > cfg["stream"]["max_result_age_seconds"]:
                stable.reset()
                sop.previous = None
                emit({"type": "prediction", "action": "unknown", "decision": "unknown",
                      "reason": "结果过期或流已重连", "latency_seconds": latency, "epoch": now_epoch})
                continue
            idx = int(probabilities.argmax())
            cls = cfg["classes"][idx]
            score = float(probabilities[idx])
            action = stable.update(cls["id"], score)
            rule = sop.update(action, end)
            emit({"type": "prediction", "candidate": cls["id"], "action": action or "unknown",
                  "name": cls["name"] if action else "无法确认", "confidence": score,
                  "decision": cls["kind"] if action else "unknown", "source_time": end,
                  "latency_seconds": latency, "epoch": window_epoch,
                  "sop": rule, "probabilities": probabilities.tolist()})
            predictions += 1
    finally:
        try:
            capture.close()
        finally:
            if handle:
                handle.close()
    return {"predictions": predictions, "seconds": time.monotonic() - start}
