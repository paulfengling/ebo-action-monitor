import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
import cv2
from .common import rows, write_rows
from .video import video_info


def resolve_video(item, manifest):
    p = Path(item["video"])
    return p if p.is_absolute() else Path(manifest).resolve().parent / p


def prepare(segments, output, manifest, reviewed=False):
    """CSV is deliberately simple: one temporal span and one label per row."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with open(segments, encoding="utf-8-sig", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            video = Path(row["video"])
            if not video.is_absolute():
                video = Path(segments).resolve().parent / video
            info = video_info(video)
            start, end = float(row["start"]), float(row["end"])
            if not 0 <= start < end <= info["seconds"] + 0.001:
                raise ValueError(f"第 {index+1} 个片段时间范围无效")
            clip = output / f"clip_{index:05d}.mp4"
            if clip.exists():
                raise ValueError("输出片段已存在，请使用新的输出目录")
            cap = cv2.VideoCapture(str(video))
            writer = None
            count = 0
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * info["fps"]))
                for _ in range(round(end * info["fps"]) - round(start * info["fps"])):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError("切片解码提前结束")
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (w, h))
                        if not writer.isOpened():
                            raise ValueError("MP4 编码器不可用")
                    writer.write(frame)
                    count += 1
            finally:
                cap.release()
                if writer:
                    writer.release()
            if count == 0:
                raise ValueError("空视频片段")
            records.append({"video": str(clip), "label": row["label"], "group_id": row["group_id"],
                            "reviewed": reviewed, "synthetic": False,
                            "source_video": str(video.resolve()), "start": start, "end": end})
    if not records:
        raise ValueError("片段表没有数据")
    write_rows(manifest, records)
    return len(records)


def validate(manifest, cfg, require_reviewed=True, allow_synthetic=False):
    items = rows(manifest)
    if not items:
        raise ValueError("数据集为空")
    labels = {c["id"] for c in cfg["classes"]}
    seen = set()
    for i, item in enumerate(items):
        if item.get("label") not in labels or not item.get("group_id"):
            raise ValueError(f"样本 {i} 的标签或 group_id 无效")
        if require_reviewed and item.get("reviewed") is not True:
            raise ValueError(f"样本 {i} 尚未人工确认标签")
        if item.get("synthetic", False) and not allow_synthetic:
            raise ValueError("拒绝将合成验证数据用于正式训练；烟雾测试需显式允许")
        video = resolve_video(item, manifest)
        info = video_info(video)
        if info["seconds"] + 0.01 < cfg["model"]["window_seconds"]:
            raise ValueError(f"样本 {i} 比模型时间窗口更短，请重新切片")
        signature = hashlib.sha256(video.read_bytes()).hexdigest()
        if signature in seen:
            raise ValueError("数据集中存在完全重复的视频")
        seen.add(signature)
        item["video"] = str(video.resolve())
    return items


def split_dataset(manifest, output, cfg, seed=42):
    items = validate(manifest, cfg)
    groups = sorted({r["group_id"] for r in items})
    if len(groups) < 3:
        raise ValueError("至少需要三个独立 group_id 才能建立 train/val/test")
    labels = {c["id"] for c in cfg["classes"]}
    rng = random.Random(seed)
    n_test = max(1, round(len(groups) * .15))
    n_val = max(1, round(len(groups) * .15))
    found = None
    for _ in range(500):
        rng.shuffle(groups)
        sets = {"test": set(groups[:n_test]), "val": set(groups[n_test:n_test+n_val]),
                "train": set(groups[n_test+n_val:])}
        result = {name: [r for r in items if r["group_id"] in subset] for name, subset in sets.items()}
        if all({r["label"] for r in subset} == labels for subset in result.values()):
            found = result
            break
    if found is None:
        raise ValueError("无法在各分组中覆盖所有动作类别；请增加独立人员/场次与缺少的类别")
    for name, subset in found.items():
        write_rows(Path(output) / f"{name}.jsonl", subset)
    return {name: dict(Counter(r["label"] for r in subset)) for name, subset in found.items()}


def verify_splits(manifests, cfg, allow_synthetic=False):
    datasets = [validate(m, cfg, allow_synthetic=allow_synthetic) for m in manifests]
    groups, hashes, spans = {}, {}, defaultdict(list)
    for split_index, dataset in enumerate(datasets):
        for row in dataset:
            group = row["group_id"]
            if group in groups and groups[group] != split_index:
                raise ValueError("数据泄漏：同一 group_id 出现在不同划分")
            groups[group] = split_index
            h = hashlib.sha256(Path(row["video"]).read_bytes()).hexdigest()
            if h in hashes and hashes[h] != split_index:
                raise ValueError("数据泄漏：同一视频内容出现在不同划分")
            hashes[h] = split_index
            if "source_video" in row and "start" in row:
                for other_split, start, end in spans[row["source_video"]]:
                    if other_split != split_index and max(start, row["start"]) < min(end, row["end"]):
                        raise ValueError("数据泄漏：不同划分的原视频时间段有重叠")
                spans[row["source_video"]].append((split_index, row["start"], row["end"]))
    return datasets
