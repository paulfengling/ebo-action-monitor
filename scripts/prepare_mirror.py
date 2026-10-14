"""Cut approximate manually observed mirror stages into EBo training clips.

This prepares a same-video demonstration, never independent train/val/test
splits. The provided timeline describes mirror_demo_01.mp4 only; supply another
timeline CSV when preparing a different video. All spans are half-open [start,end).
"""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path

import yaml

from ebo.common import config, rows, save_json, write_rows
from ebo.data import prepare, validate
from ebo.video import video_info


# Boundaries were estimated by human observation of sampled frames, not by a
# trained model and not by a process engineer. End=None uses the decoded duration.
MIRROR_TIMELINE = [
    (0.0, 2.0, "background", "背景或尚未开始"),
    (2.0, 34.0, "base_connect", "S1 底座与立杆连接"),
    (34.0, 47.0, "mirror_align", "S2 镜面与支架对齐"),
    (47.0, 55.0, "fasten_side_a", "S3 第一次侧边紧固，称侧 A"),
    (55.0, 69.0, "fasten_side_b", "S4 第二次侧边紧固，称侧 B"),
    (69.0, 91.0, "final_check", "S5 抬起、翻转、调整与整体检查"),
    (91.0, None, "background", "背景或镜头调整"),
]


def _write_csv(path, fields, records):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def prepare_mirror(video, output, config_path="config/mirror.yaml", timeline=None,
                   clip_seconds=2.0, group_id=None):
    video, output = Path(video).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("输出路径已经存在；请使用新的 --out，避免覆盖已有片段")
    if not video.is_file():
        raise ValueError("输入视频不存在")
    if timeline is None and video.name != "mirror_demo_01.mp4":
        raise ValueError("默认时间线仅适用于 mirror_demo_01.mp4；其他视频请指定 --timeline")
    if not math.isfinite(clip_seconds) or clip_seconds <= 0:
        raise ValueError("clip_seconds 必须为有限正数")
    cfg, info = config(config_path), video_info(video)
    fps, nframes = info["fps"], info["frames"]
    min_frames = math.ceil(cfg["model"]["window_seconds"] * fps - 1e-8)
    target_frames = max(min_frames, round(clip_seconds * fps))
    group_id = group_id or video.stem
    if not group_id.strip():
        raise ValueError("group_id 不能为空")
    if timeline:
        with open(timeline, encoding="utf-8-sig", newline="") as source:
            spans = [(float(r["start"]), float(r["end"]), r["label"],
                      r.get("description", r["label"])) for r in csv.DictReader(source)]
    else:
        spans = [(a, info["seconds"] if b is None else b, label, description)
                 for a, b, label, description in MIRROR_TIMELINE]
    labels = {c["id"] for c in cfg["classes"]}
    timeline_rows, segment_rows = [], []
    previous_end = 0
    for start, end, label, description in spans:
        if (not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end <= info["seconds"] + .001
                or label not in labels):
            raise ValueError("时间线的时间范围或 label 无效")
        begin_frame, end_frame = round(start * fps), min(nframes, round(end * fps))
        if begin_frame != previous_end:
            raise ValueError("时间线必须按顺序连续覆盖视频，不允许重叠或遗漏")
        length = end_frame - begin_frame
        if length < min_frames:
            raise ValueError(f"动作 {label} 短于模型时间窗口，请合并或修订边界")
        previous_end = end_frame
        timeline_rows.append({"start": begin_frame / fps, "end": end_frame / fps,
                              "label": label, "description": description,
                              "first_frame": begin_frame, "end_frame_exclusive": end_frame,
                              "observed_start_seconds": start, "observed_end_seconds": end,
                              "annotation_source": "manual_observation",
                              "boundary_precision": "approximate"})
        # Partition by integer frame counts. Redistribute a short remainder so
        # every clip remains at least one complete model window, with no overlap.
        count = max(1, length // target_frames)
        while count > 1 and length // count < min_frames:
            count -= 1
        for index in range(count):
            first = begin_frame + index * length // count
            last = begin_frame + (index + 1) * length // count
            segment_rows.append({"video": str(video), "start": first / fps,
                                 "end": last / fps, "label": label, "group_id": group_id})
    if not spans or previous_end != nframes:
        raise ValueError("时间线必须覆盖整个视频")
    if {r["label"] for r in timeline_rows} != labels:
        raise ValueError("时间线没有覆盖配置中的所有类别")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                        encoding="utf-8")
    _write_csv(output / "timeline.csv", list(timeline_rows[0]), timeline_rows)
    _write_csv(output / "segments.csv", list(segment_rows[0]), segment_rows)
    manifest = output / "reviewed.jsonl"
    prepare(output / "segments.csv", output / "clips", manifest, reviewed=True)
    records = rows(manifest)
    for item in records:
        item.update(annotation_source="manual_observation", boundary_precision="approximate",
                    demonstration_only=True, independent_evaluation=False,
                    review_note="人工画面观察的动作阶段标签；不是工艺合格标注，边界需现场复核")
    write_rows(manifest, records)
    checked = validate(manifest, cfg)
    summary = {"source_video": str(video), "video": info, "group_id": group_id,
               "manifest": str(manifest), "clips": len(checked),
               "class_counts": dict(Counter(r["label"] for r in checked)),
               "min_clip_seconds": min(video_info(r["video"])["seconds"] for r in checked),
               "annotation_source": "manual_observation", "boundary_precision": "approximate",
               "demonstration_only": True, "independent_evaluation": False,
               "notes": ["单次单视频数据，禁止拆分后声称独立测试准确率。",
                         "时间边界为人工观察估计，时间线已对齐视频帧。",
                         "S3/S4 仅表示本次演示先后两侧，不保证跨视频侧别识别。",
                         "只标注正常动作阶段，未提供漏装、假拧等质量不良样本。"]}
    save_json(output / "preparation.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="mirror_demo_01.mp4")
    parser.add_argument("--out", default="runs/mirror_ebo/input")
    parser.add_argument("--config", default="config/mirror.yaml")
    parser.add_argument("--timeline", help="替代时间线 CSV：start,end,label,description，连续覆盖全片")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--group-id")
    args = parser.parse_args()
    summary = prepare_mirror(args.video, args.out, args.config, args.timeline,
                             args.clip_seconds, args.group_id)
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
