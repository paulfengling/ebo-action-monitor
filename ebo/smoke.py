import copy
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from .common import write_rows, save_json
from .training import train, evaluate
from .inference import run


def smoke(output, base_cfg, pretrained=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("烟雾测试目录非空，请使用新目录")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = copy.deepcopy(base_cfg)
    cfg["model"].update(frames=4, window_seconds=1.0, crop=[32, 32], resize=[40, 56])
    cfg["train"].update(epochs=1, batch_size=2, freeze_backbone=True, workers=0)
    cfg["stream"].update(stable_predictions=1, max_result_age_seconds=5)
    cfg["sop"]["enabled"] = False
    import yaml
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    manifests = {}
    for split, split_index in (("train", 0), ("val", 1), ("test", 2)):
        records = []
        for index, cls in enumerate(cfg["classes"]):
            path = output / f"{split}_{index}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (80, 64))
            if not writer.isOpened():
                raise RuntimeError("烟雾测试 MP4 编码失败")
            for t in range(20):
                frame = np.full((64, 80, 3), 25+index*35+split_index*3, dtype=np.uint8)
                cv2.rectangle(frame, (5+t, 10+index), (15+t, 30+index), (30, 180, 220), -1)
                writer.write(frame)
            writer.release()
            records.append({"video": str(path), "label": cls["id"], "group_id": split,
                            "reviewed": True, "synthetic": True})
        manifests[split] = output / f"{split}.jsonl"
        write_rows(manifests[split], records)
    checkpoint = train(cfg, manifests["train"], manifests["val"], output / "model", "cpu", pretrained, True)
    report = evaluate(checkpoint, manifests["test"], output / "evaluation.json", "cpu", True,
                      manifests["train"], manifests["val"])
    events = []
    inference = run(checkpoint, str(output / "test_0.mp4"), cfg, "cpu", str(output / "inference.jsonl"),
                    allow_demo=True, on_result=events.append)
    if inference["predictions"] < 1:
        raise RuntimeError("端到端测试未产生视频预测")
    try:
        run(checkpoint, str(output / "test_0.mp4"), cfg, "cpu", allow_demo=False)
        raise AssertionError("合成模型阻止机制未生效")
    except ValueError:
        pass
    result = {"pipeline_verified": True, "synthetic_only": True, "pretrained_download_verified": pretrained,
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "inference": inference, "test_samples": sum(c["support"] for c in report["per_class"]),
              "note": "只验证代码链路，不代表真实动作精度或 RTSP 现场延迟"}
    save_json(output / "verification.json", result)
    return result
