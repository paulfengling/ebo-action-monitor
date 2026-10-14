"""Single-source fitted replay. This does not bypass production train/val safeguards."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import save_json
from .data import validate
from .model import make_model, transform, tensor, device
from .video import read_window


def absorb_standardization(head, mean, std):
    """Keep checkpoints compatible with ordinary R3D's unmodified linear fc."""
    result = nn.Linear(head.in_features, head.out_features)
    with torch.no_grad():
        weights = head.weight / std.unsqueeze(0)
        result.weight.copy_(weights)
        result.bias.copy_(head.bias - (weights * mean.unsqueeze(0)).sum(1))
    return result


def fit_replay(cfg, manifest, out, device_name="auto", epochs=1200, learning_rate=.02):
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("训练轮数与学习率必须大于零")
    records = validate(manifest, cfg)
    if len({r["group_id"] for r in records}) != 1:
        raise ValueError("fit-replay 仅用于单来源拟合演示；多个独立来源请使用 train")
    class_ids = [c["id"] for c in cfg["classes"]]
    if set(r["label"] for r in records) != set(class_ids):
        raise ValueError("拟合数据需覆盖全部配置类别")
    source_paths = {r.get("source_video") for r in records}
    if None in source_paths or len(source_paths) != 1:
        raise ValueError("拟合演示要求所有片段具有同一 source_video 来源")
    output = Path(out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，请使用新目录，避免覆盖模型")
    output.mkdir(parents=True, exist_ok=True)
    target = device(device_name)
    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.monotonic()
    backbone = make_model(len(class_ids), pretrained=True)
    backbone.fc = nn.Identity()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone = backbone.to(target).eval()
    processor = transform(cfg["model"])
    features = []
    targets = []
    with torch.inference_mode():
        for index, row in enumerate(records):
            x = tensor(read_window(row["video"], cfg["model"]), processor, cfg["model"].get("roi"))
            features.append(backbone(x.unsqueeze(0).to(target)).cpu().clone())
            targets.append(class_ids.index(row["label"]))
            if (index+1) % 10 == 0 or index+1 == len(records):
                print(f"预训练特征：{index+1}/{len(records)}", flush=True)
    # Train only the linear head on CPU. No validation score or early-stop on test data.
    x = torch.cat(features).clone().detach()
    y = torch.tensor(targets, dtype=torch.long)
    mean, std = x.mean(0), x.std(0).clamp_min(.001)
    standardized = (x-mean)/std
    head = nn.Linear(x.shape[1], len(class_ids))
    counts = torch.bincount(y, minlength=len(class_ids)).float()
    weights = (counts.sum()/counts).div(len(class_ids))
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.001)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = head(standardized)
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_value = float(loss)
            accuracy = float((head(standardized).argmax(1)==y).float().mean())
        history.append({"epoch": epoch+1, "training_loss": loss_value,
                        "training_fit_accuracy": accuracy})
        if (epoch+1) % 100 == 0:
            print(json.dumps(history[-1]), flush=True)
        if epoch >= 99 and accuracy == 1 and loss_value < .02:
            break
    final_head = absorb_standardization(head, mean, std)
    with torch.no_grad():
        equivalence_error = float((head(standardized)-final_head(x)).abs().max())
    if equivalence_error > .002:
        raise RuntimeError("分类头标准化合并误差过大")
    backbone = backbone.cpu()
    backbone.fc = final_head
    original = Path(next(iter(source_paths))).resolve()
    provenance = {"evidence_scope": "same_video_replay", "calibration_only": True,
                  "generalization_validated": False, "quality_assessment": "not_available",
                  "source_video_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                  "labels_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                  "samples": len(records), "class_counts": dict(Counter(r["label"] for r in records)),
                  "label_source": "manual_observation_approximate_boundaries",
                  "fit_metrics": history[-1], "head_equivalence_max_error": equivalence_error,
                  "seconds": time.monotonic()-started,
                  "note": "同一视频标签拟合与回放，不是独立验证；不代表装配质量正确"}
    checkpoint = {"schema_version": 1, "config": cfg, "state_dict": backbone.state_dict(),
                  "demo_only": True, "pretrained_origin": True, "epoch": len(history),
                  "history": history, **provenance}
    torch.save(checkpoint, output/"replay_model.pt")
    save_json(output/"calibration.json", provenance)
    save_json(output/"history.json", history)
    np.savez_compressed(output/"features.npz", features=x.numpy(), labels=y.numpy())
    return {"checkpoint": str(output/"replay_model.pt"), **provenance}
