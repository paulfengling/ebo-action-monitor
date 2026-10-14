import json
import os
from pathlib import Path
import yaml


def config(path="config/project.yaml"):
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes = obj["classes"]
    ids = [c["id"] for c in classes]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("至少两个动作类别，且类别 id 不能重复")
    if any(c["kind"] not in ("correct", "incorrect", "neutral") for c in classes):
        raise ValueError("kind 必须为 correct / incorrect / neutral")
    m, s = obj["model"], obj["stream"]
    if m["architecture"] != "r3d_18" or m["frames"] < 4 or m["window_seconds"] <= 0:
        raise ValueError("无效模型或时间窗口配置")
    if len(m["crop"]) != 2 or len(m["resize"]) != 2 or any(v <= 0 for v in m["crop"] + m["resize"]):
        raise ValueError("无效图像尺寸")
    roi = m.get("roi")
    if roi is not None and (len(roi) != 4 or not all(0 <= v <= 1 for v in roi) or not roi[0] < roi[2] or not roi[1] < roi[3]):
        raise ValueError("ROI 必须是0–1范围的有效矩形")
    if not 0 <= s["confidence_threshold"] <= 1 or s["stable_predictions"] < 1:
        raise ValueError("无效置信度或稳定化配置")
    if any(s[k] <= 0 for k in ("open_timeout_ms", "read_timeout_ms", "reconnect_seconds",
                               "max_gap_seconds", "inference_interval", "max_result_age_seconds")):
        raise ValueError("流超时、重连间隔、预测间隔必须大于零")
    if any(obj["train"][k] <= 0 for k in ("epochs", "batch_size", "learning_rate")) or obj["train"]["workers"] < 0:
        raise ValueError("无效训练参数")
    if any(step["action"] not in ids or step["min_seconds"] < 0 for step in obj["sop"]["steps"]):
        raise ValueError("SOP 动作或时间配置无效")
    return obj


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_env(path=".env"):
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def source_value(source, cfg):
    if source is not None:
        return int(source) if source.isdigit() else source
    value = os.getenv(cfg["stream"]["source_env"])
    if not value or "CAMERA_IP" in value:
        raise ValueError("请在 .env 设置 RTSP_URL，或通过 --source 指定本地视频")
    return value
