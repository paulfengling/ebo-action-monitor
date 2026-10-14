import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.video import r3d_18, R3D_18_Weights
from .video import read_window, video_info


def device(name="auto"):
    if name == "auto":
        # CUDA is the production target; CPU is a portable verification path.
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用 CUDA，请检查驱动与 CUDA 版 PyTorch")
    return torch.device(name)


def make_model(classes, pretrained=False):
    model = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def transform(model):
    return R3D_18_Weights.DEFAULT.transforms(crop_size=model["crop"], resize_size=model["resize"])


def tensor(frames, processor, roi=None):
    # Capture arrays are BGR; pretrained transforms expect RGB, T C H W.
    if roi is not None:
        cropped = []
        for frame in frames:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = roi
            image = frame[int(y1*h):max(int(y1*h)+1, int(y2*h)), int(x1*w):max(int(x1*w)+1, int(x2*w))]
            if image.size == 0:
                raise ValueError("ROI 裁剪区域为空")
            cropped.append(image)
        frames = cropped
    array = np.stack([f[:, :, ::-1].copy() for f in frames])
    return processor(torch.from_numpy(array).permute(0, 3, 1, 2))


class Clips(Dataset):
    def __init__(self, records, cfg, training=False):
        self.records, self.cfg, self.training = records, cfg, training
        self.ids = {c["id"]: i for i, c in enumerate(cfg["classes"])}
        self.processor = transform(cfg["model"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        start = None
        if self.training:
            duration = video_info(row["video"])["seconds"]
            start = random.uniform(0, max(0, duration - self.cfg["model"]["window_seconds"]))
        return tensor(read_window(row["video"], self.cfg["model"], start), self.processor, self.cfg["model"].get("roi")), self.ids[row["label"]]


def load_checkpoint(path, target_device, allow_demo=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("schema_version") != 1:
        raise ValueError("不支持的 checkpoint 格式")
    if ckpt.get("demo_only") and not allow_demo:
        raise ValueError("这是合成数据烟雾测试模型，禁止正式推理；测试时显式 --allow-demo-model")
    model = make_model(len(ckpt["config"]["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(target_device).eval(), ckpt
