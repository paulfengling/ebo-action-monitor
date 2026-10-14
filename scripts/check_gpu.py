"""Run on deployment host: compute sanity and peak memory for one default window."""
import json
import time
import torch
from torchvision.models.video import r3d_18

if not torch.cuda.is_available():
    raise SystemExit("没有可用 CUDA；请在 NVIDIA 部署机安装匹配的 PyTorch")
model = r3d_18(weights=None).eval().cuda()
x = torch.zeros((1, 3, 16, 112, 112), device="cuda")
with torch.inference_mode():
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()
print(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "mean_forward_seconds": (time.perf_counter()-started)/20,
                  "peak_allocated_MB": torch.cuda.max_memory_allocated()/1024**2,
                  "note": "仅网络前向；不含 RTSP、预处理、训练与动作精度"}, ensure_ascii=False, indent=2))
