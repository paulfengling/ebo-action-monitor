# 本地部署与 RTX 5080 配置

## 已有 RTX 5080

RTX 5080 为 16GB 显存。优先使用现有卡验证本项目 R3D-18 训练和一路 RTSP 推理，不需要先换更贵 GPU。batch_size=2 是起始配置；显存峰值、速度和热稳定性要在目标机器实测。

建议整机：

| 部件 | 配置建议 |
|---|---|
| GPU | 已有 RTX 5080 16GB |
| CPU | 12–16 个物理核心，兼顾视频解码和数据加载 |
| RAM | 64GB；大量录像/多进程训练可升级 128GB |
| 系统/模型盘 | 2TB NVMe SSD |
| 数据盘 | 额外 4TB，按录像码率与保留天数调整 |
| 电源 | 正规整机 850–1000W 级，按具体 CPU/GPU 与厂商规范核定 |
| 操作系统 | Ubuntu 24.04 LTS 优先，训练/推理使用 NVIDIA CUDA |

如果后续需要同时常驻较大的视觉语言模型，可考虑 RTX 5090 32GB 或 48GB 专业卡，重新压测。当前无需因为“大模型”三个字直接采购多卡服务器。

来源：[RTX 5080 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)、[PyTorch 安装选择器](https://pytorch.org/get-started/locally/)。上述整机是工程配置建议，非官方现场性能保证。

## NVIDIA 软件安装

先安装支持 Blackwell 的 NVIDIA 驱动，并执行 nvidia-smi。按 PyTorch 官方当前安装选择器选择 Linux/Pip/Python/CUDA，安装匹配的 torch 与 torchvision；不要照搬旧 CUDA 11.8 命令。

随后在同一环境：

```bash
python -m pip install -e '.[ui]'
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
python -m ebo.cli train --device cuda
```

该主机没有 NVIDIA GPU，GPU 命令与显存容量尚未本机实测。PyQt 在无桌面的 Linux 服务器上应使用 infer CLI；monitor/annotate 需要可用桌面显示。Mac 的 CPU 路线可验证代码，但不代表 GPU 部署速度。

正式模型的类别与预处理必须和运行配置一致。默认推理读取本地 best.pt，不调用云端。

## 可选本地大模型标注服务

建议先使用 Qwen3-VL-4B-Instruct，为 RTX 5080 控制图像数和上下文；不同时运行动作训练和大模型服务。16GB 的真实余量取决于权重、视觉编码、KV cache 和运行框架，以下参数仅为待验证模板。

单独创建 VLM Python 环境，按 vLLM 的 Blackwell GPU 安装文档配置驱动、CUDA 与依赖，提前下载权重/tokenizer/processor。服务环境与动作模型环境分开，避免 torch 依赖互相覆盖。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm serve /opt/models/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":8,"video":0}'
```

如果启动或图像请求超显存，降低视觉像素预算/使用经验证的量化 checkpoint；不要默认 BF16 模板必然适配。标注服务端也可能因8帧图像提示超过8192 token 而拒绝请求，需按图像 token 预算调整上下文或服务端分辨率。

填入 .env：

```text
VLM_URL=http://127.0.0.1:8000/v1/chat/completions
VLM_MODEL=Qwen3-VL-4B-Instruct
```

服务启动后执行 propose，再用 annotate 人工确认。这里只提供请求协议和服务模板，没有将 VLM 微调作为在线模型，也没有声称真实 Qwen 服务已部署通过。

来源：[Qwen3-VL 官方项目](https://github.com/QwenLM/Qwen3-VL)、[vLLM GPU 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)。

## 上线前检查

- 连续运行真实 RTSP，测试断线重连、停流、分辨率变化与关闭。
- 确认缓冲延迟；本机接收时间无法发现摄像头内部积压。
- 实测推理计算时间、完整动作提醒延迟、显存峰值、温度和日志增长。
- 完整流程录像验收，不只看单片段 accuracy。
- 当前会话分割需人工重启推理。多人/连续多员工需要增加人员检测、跟踪与会话管理后才可无人值守。
