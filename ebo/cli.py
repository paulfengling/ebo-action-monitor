import argparse
import json
import os
import sys
from pathlib import Path
from .common import config, load_env, source_value, write_rows


def parser():
    p = argparse.ArgumentParser(description="EBo 视频动作训练与 RTSP 推理")
    p.add_argument("--config", default="config/project.yaml")
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="检查运行依赖、FFmpeg 与 CUDA，不打印摄像头凭据")
    rec = subs.add_parser("record", help="RTSP/视频分段录制")
    rec.add_argument("--source")
    rec.add_argument("--out", required=True)
    rec.add_argument("--seconds", type=float, default=60)
    rec.add_argument("--clip-seconds", type=float, default=10)
    rec.add_argument("--fps", type=float, default=15)
    prep = subs.add_parser("prepare", help="按人工时间段表切片")
    prep.add_argument("--segments", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--reviewed", action="store_true", help="仅当 CSV 标签已经人工确认时使用")
    imp = subs.add_parser("import-videos", help="生成待标注清单")
    imp.add_argument("--input", required=True)
    imp.add_argument("--group-id", required=True)
    imp.add_argument("--out", required=True)
    ann = subs.add_parser("annotate", help="PyQt 片段标注/候选标签人工确认")
    ann.add_argument("--manifest")
    ann.add_argument("--out", default="data/labels/reviewed.jsonl")
    propose = subs.add_parser("propose", help="本地大模型辅助标注，输出均为待确认")
    propose.add_argument("--manifest", required=True)
    propose.add_argument("--out", required=True)
    propose.add_argument("--endpoint", default=os.getenv("VLM_URL"))
    propose.add_argument("--model", default=os.getenv("VLM_MODEL"))
    for cmd in ("validate", "split"):
        s = subs.add_parser(cmd)
        s.add_argument("--manifest", required=True)
        if cmd == "split":
            s.add_argument("--out", default="data/splits")
    train = subs.add_parser("train")
    train.add_argument("--train", default="data/splits/train.jsonl")
    train.add_argument("--val", default="data/splits/val.jsonl")
    train.add_argument("--out", default="runs/action_model")
    train.add_argument("--device", default="auto")
    train.add_argument("--from-scratch", action="store_true")
    train.add_argument("--allow-synthetic", action="store_true", help="仅烟雾测试")
    train.add_argument("--resume")
    evaluation = subs.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--test", default="data/splits/test.jsonl")
    evaluation.add_argument("--train", default="data/splits/train.jsonl")
    evaluation.add_argument("--val", default="data/splits/val.jsonl")
    evaluation.add_argument("--out", default="runs/evaluation.json")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--allow-demo-model", action="store_true")
    for cmd in ("infer", "monitor"):
        inf = subs.add_parser(cmd)
        inf.add_argument("--checkpoint", required=True)
        inf.add_argument("--source")
        inf.add_argument("--device", default="auto")
        inf.add_argument("--log", default="runs/inference.jsonl")
        inf.add_argument("--duration", type=float)
        inf.add_argument("--allow-demo-model", action="store_true")
    smoke = subs.add_parser("smoke", help="合成数据端到端验证；不是现场训练")
    smoke.add_argument("--out", default="runs/smoke")
    smoke.add_argument("--pretrained", action="store_true", help="下载并验证官方预训练权重")
    return p


def main(argv=None):
    load_env()
    args = parser().parse_args(argv)
    try:
        cfg = config(args.config)
        if args.command == "doctor":
            import cv2
            import torch
            import torchvision
            result = {"python": sys.version.split()[0], "torch": torch.__version__,
                      "torchvision": torchvision.__version__, "opencv": cv2.__version__,
                      "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
                      "ffmpeg_backend": cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG),
                      "rtsp_configured": bool(os.getenv(cfg["stream"]["source_env"]))}
            if torch.cuda.is_available():
                result.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
        elif args.command == "record":
            from .recording import record
            result = record(source_value(args.source, cfg), cfg, args.out, args.seconds, args.clip_seconds, args.fps)
        elif args.command == "prepare":
            from .data import prepare
            result = {"clips": prepare(args.segments, args.out, args.manifest, args.reviewed)}
        elif args.command == "import-videos":
            records = [{"video": str(p.resolve()), "group_id": args.group_id, "reviewed": False,
                        "label": "unknown", "synthetic": False} for p in sorted(Path(args.input).glob("*"))
                       if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
            if not records:
                raise ValueError("目录中没有视频")
            write_rows(args.out, records)
            result = {"pending_clips": len(records)}
        elif args.command == "annotate":
            from .ui import annotate
            return annotate(cfg, args.out, args.manifest)
        elif args.command == "propose":
            from .vlm import propose
            result = {"proposals": propose(args.manifest, args.out, cfg, args.endpoint, args.model)}
        elif args.command == "validate":
            from .data import validate
            result = {"reviewed_clips": len(validate(args.manifest, cfg))}
        elif args.command == "split":
            from .data import split_dataset
            result = split_dataset(args.manifest, args.out, cfg, cfg["train"]["seed"])
        elif args.command == "train":
            from .training import train
            result = {"checkpoint": train(cfg, args.train, args.val, args.out, args.device,
                                           not args.from_scratch, args.allow_synthetic, args.resume)}
        elif args.command == "evaluate":
            from .training import evaluate
            result = evaluate(args.checkpoint, args.test, args.out, args.device, args.allow_demo_model,
                              args.train, args.val)
        elif args.command in ("infer", "monitor"):
            options = {"checkpoint": args.checkpoint, "source": source_value(args.source, cfg),
                       "runtime_cfg": cfg, "device_name": args.device, "log": args.log,
                       "duration": args.duration, "allow_demo": args.allow_demo_model}
            if args.command == "monitor":
                from .ui import monitor
                return monitor(options)
            from .inference import run
            result = run(**options)
        else:
            from .smoke import smoke
            result = smoke(args.out, cfg, args.pretrained)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        # Inputs can contain credentials; only trusted validation messages are emitted.
        if args.command in ("infer", "monitor", "record"):
            print("失败：请检查配置、模型和视频源。错误类型：" + type(exc).__name__, file=sys.stderr)
        else:
            print("失败：" + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
