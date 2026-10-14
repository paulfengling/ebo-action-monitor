import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import save_json
from .data import verify_splits, validate
from .model import Clips, make_model, device, load_checkpoint


def metrics(targets, predictions, classes):
    matrix = np.zeros((classes, classes), dtype=int)
    for actual, predicted in zip(targets, predictions):
        matrix[actual, predicted] += 1
    stats = []
    for i in range(classes):
        tp = int(matrix[i, i])
        precision = tp / max(1, int(matrix[:, i].sum()))
        recall = tp / max(1, int(matrix[i].sum()))
        stats.append({"precision": precision, "recall": recall,
                      "f1": 2*precision*recall/max(1e-12, precision+recall), "support": int(matrix[i].sum())})
    return {"accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
            "macro_f1": float(np.mean([s["f1"] for s in stats])),
            "per_class": stats, "confusion_matrix": matrix.tolist()}


@torch.inference_mode()
def assess(model, loader, target_device, nclasses):
    model.eval()
    targets, predictions, probabilities = [], [], []
    for x, y in loader:
        probs = model(x.to(target_device)).softmax(1).cpu()
        targets.extend(y.tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())
    return metrics(targets, predictions, nclasses), targets, probabilities


def train(cfg, train_manifest, val_manifest, out, device_name="auto", pretrained=True,
          allow_synthetic=False, resume=None):
    records, val_records = verify_splits([train_manifest, val_manifest], cfg, allow_synthetic)
    labels = {c["id"] for c in cfg["classes"]}
    if any({r["label"] for r in subset} != labels for subset in (records, val_records)):
        raise ValueError("训练集和验证集均须包含所有类别")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "last.pt").exists():
        raise ValueError("输出目录已有模型，请更换目录或使用 --resume")
    t = cfg["train"]
    random.seed(t["seed"])
    np.random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    target_device = device(device_name)
    model = make_model(len(labels), pretrained and not resume).to(target_device)
    if t["freeze_backbone"]:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=t["learning_rate"])
    loader = DataLoader(Clips(records, cfg, True), batch_size=t["batch_size"], shuffle=True,
                        num_workers=t["workers"])
    val_loader = DataLoader(Clips(val_records, cfg), batch_size=t["batch_size"], num_workers=t["workers"])
    history, best, start_epoch = [], -1.0, 0
    pretrained_origin = pretrained
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["config"]["classes"] != cfg["classes"] or saved["config"]["model"] != cfg["model"]:
            raise ValueError("续训的类别或模型配置与原 checkpoint 不符")
        if saved["config"]["train"]["freeze_backbone"] != t["freeze_backbone"]:
            raise ValueError("续训不能改变 freeze_backbone")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, history, best = saved["epoch"], saved["history"], saved["best_macro_f1"]
        pretrained_origin = saved["pretrained_origin"]
        if not (output / "best.pt").exists():
            # A new output directory cannot reuse an unseen historic "best" checkpoint.
            best = -1.0
    if t["epochs"] <= start_epoch:
        raise ValueError("epochs 必须大于已完成的训练轮数")
    fingerprints = {"train": hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest(),
                    "val": hashlib.sha256(Path(val_manifest).read_bytes()).hexdigest()}
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        if t["freeze_backbone"]:
            model.eval()  # Do not update backbone BatchNorm while training only the head.
            model.fc.train()
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(target_device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(target_device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        result, _, _ = assess(model, val_loader, target_device, len(labels))
        result.update(epoch=epoch+1, train_loss=total_loss / len(records))
        history.append(result)
        improved = result["macro_f1"] > best
        best = max(best, result["macro_f1"])
        checkpoint = {"schema_version": 1, "config": cfg, "state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "epoch": epoch+1, "history": history,
                      "best_macro_f1": best, "manifest_hashes": fingerprints,
                      "pretrained_origin": pretrained_origin,
                      "demo_only": any(r.get("synthetic") for r in records + val_records)}
        torch.save(checkpoint, output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        save_json(output / "history.json", history)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return str(output / "best.pt")


def evaluate(checkpoint, manifest, out, device_name="auto", allow_demo=False,
             train_manifest=None, val_manifest=None):
    target_device = device(device_name)
    model, saved = load_checkpoint(checkpoint, target_device, allow_demo)
    cfg = saved["config"]
    if train_manifest and val_manifest:
        records = verify_splits([train_manifest, val_manifest, manifest], cfg, allow_demo)[2]
    else:
        records = validate(manifest, cfg, allow_synthetic=allow_demo)
    loader = DataLoader(Clips(records, cfg), batch_size=cfg["train"]["batch_size"])
    result, targets, probabilities = assess(model, loader, target_device, len(cfg["classes"]))
    threshold = cfg["stream"]["confidence_threshold"]
    correct = {i for i, c in enumerate(cfg["classes"]) if c["kind"] == "correct"}
    incorrect = {i for i, c in enumerate(cfg["classes"]) if c["kind"] == "incorrect"}
    false_pass = sum(y in incorrect and int(np.argmax(p)) in correct and max(p) >= threshold
                     for y, p in zip(targets, probabilities))
    n_incorrect = sum(y in incorrect for y in targets)
    result.update(classes=cfg["classes"], demo_only=saved["demo_only"],
                  unknown_rate=sum(max(p) < threshold for p in probabilities) / len(probabilities),
                  incorrect_as_correct_rate=false_pass / n_incorrect if n_incorrect else None,
                  split_leakage_checked=bool(train_manifest and val_manifest))
    save_json(out, result)
    return result
