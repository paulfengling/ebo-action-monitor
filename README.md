# EBo 视频动作训练与 RTSP 监测

本项目实现 **RTSP 采集 → 视频片段标注 → 数据检查/划分 → 预训练视频动作模型微调 → 独立评估 → RTSP 实时推理**。

采用“大模型辅助标注 + 专用动作模型持续推理”的路线。可选本地 Qwen-VL 服务生成待确认标签；在线模型为 torchvision 的 Kinetics400 预训练 R3D-18。本项目没有声称已经训练出识别现场动作的模型。

## 先运行验证

克隆仓库并建立虚拟环境：

```bash
git clone https://github.com/paulfengling/ebo-action-monitor.git
cd ebo-action-monitor
python3 -m venv .venv
source .venv/bin/activate
# NVIDIA 机器先按 PyTorch 官方安装选择器安装适合显卡的 CUDA 版 torch/torchvision。
python -m pip install -e '.[ui]'
python -m ebo.cli --help
python -m ebo.cli doctor
python -m unittest discover -s tests -v
```

端到端烟雾测试（输出目录必须为空）：

```bash
python -m ebo.cli smoke --out runs/smoke_new --pretrained
```

它会生成合成视频、下载官方预训练权重、执行一轮训练、保存 checkpoint、评估和推理；生成 `verification.json`。烟雾测试使用小尺寸和较短窗口以验证代码，不验证现场精度。checkpoint 标记 `demo_only=true`，不允许正式推理。

## 1. 确认要识别的动作

修改 [config/project.yaml](config/project.yaml) 的 `classes`。当前四类为示例：等待、拿工具、执行操作、错误操作。`id` 是稳定标识，`name` 是中文显示，`kind` 为 `correct / incorrect / neutral`。

不要把不同错误都笼统混在一类；正式项目应给每种可观察动作写清定义及易混淆反例。如果要求理解工具与位置，可分别定义“工具A处理位置X”等具体动作类别。此版本是单人固定工位的视频片段分类，不含多人关联、工具检测框或精确次数计数。

默认配置：2 秒窗口、16 帧，约每 0.5 秒尝试预测；输入先缩放再中心裁剪至 112×112。它是可运行基线，小工具、手部细节或左右部位容易丢失，需现场验证并考虑专门 ROI/更高分辨率模型。

`model.roi` 可配置固定人员/操作区域，如 `[0.2, 0.1, 0.8, 0.95]`，坐标范围0–1；训练与推理会先裁剪该区域，再做统一预处理。它是固定区域，不是自动人员检测。

## 2. 通过 RTSP 采集视频

```bash
cp .env.example .env
```

在 `.env` 填写自己的 `RTSP_URL`。密码含 `@`、`#` 等字符时，应按 URL 规则编码。真实地址不要写入命令行历史或提交 Git。

```bash
python -m ebo.cli record --out data/raw/operator01_session01 --seconds 600 --clip-seconds 30 --fps 15
```

录像由接收时钟重采样为固定帧率，原始分辨率保留；会切分为 MP4 并写 `recordings.jsonl`。断线关闭当前文件，重连新开文件，避免跨断线录像。该录制时间不是摄像头硬件时间戳。

也可用本地文件验证：

```bash
python -m ebo.cli record --source /absolute/path/example.mp4 --out data/raw/file_test --seconds 10
```

## 3. 标注片段：只需片段标签，不需逐帧框

打开 PyQt 标注界面，添加视频、播放、填写起止秒数、选择动作、填写 `group_id`，点击“切片并人工确认标签”：

```bash
python -m ebo.cli annotate --out data/labels/reviewed.jsonl
```

每次确认都会生成一段 MP4 和一条 `reviewed=true` 记录。片段至少与模型时间窗口一样长；一段只描述一个主要动作。更多样本规范见 [数据与训练说明](docs/DATA_AND_TRAINING.md)。

批量 CSV 切片也可使用：

```bash
python -m ebo.cli prepare --segments examples/segments.csv --out data/clips/batch01 --manifest data/labels/batch01.jsonl --reviewed
```

CSV 的 `video` 相对 CSV 所在目录解析；示例中的文件不存在，需替换实际录像与时间段。`--reviewed` 仅用于已人工确认的 CSV。没有它，输出不能直接训练。

### 可选：本地大模型辅助标注

先将录像切成单动作片段，再生成待标注清单：

```bash
python -m ebo.cli import-videos --input data/clips/unlabeled --group-id operator01 --out data/labels/pending.jsonl
python -m ebo.cli propose --manifest data/labels/pending.jsonl --out data/labels/proposals.jsonl
python -m ebo.cli annotate --manifest data/labels/proposals.jsonl --out data/labels/reviewed.jsonl
```

`propose` 读取 `.env` 的 `VLM_URL`、`VLM_MODEL`，每个片段发送 8 帧。输出只增加候选标签与理由，永远标为待人工确认。提供兼容本地服务接口，不会自动启动或下载大模型；部署模板见 [部署说明](docs/DEPLOYMENT.md)。

## 4. 检查和划分数据集

```bash
python -m ebo.cli validate --manifest data/labels/reviewed.jsonl
python -m ebo.cli split --manifest data/labels/reviewed.jsonl --out data/splits
```

按 `group_id` 划分约 70%/15%/15%。建议用人员 ID，使同一人员的全部录像在同一划分；如果按场次分组，测试不能证明对新人员泛化。至少三个独立组，各划分必须包含全部类别。程序检查重复视频、未确认标签和视频时长；训练/评估还检查跨划分的分组、重复文件与原录像重叠时间段。

## 5. 训练 / 续训

在 `config/project.yaml` 修改 epochs、batch size、learning rate 等。RTX 5080 可从 batch size 2 开始实测。

```bash
python -m ebo.cli train --device cuda --out runs/action_model
```

默认下载 Kinetics400 预训练权重并微调整个 R3D-18。`freeze_backbone: true` 可先训练分类头。输出 `best.pt`（按验证集 macro-F1）、`last.pt`、`history.json`；checkpoint 含类别、预处理、优化器、数据清单哈希和模型来源。

续训时提高配置中的总 epochs：

```bash
python -m ebo.cli train --device cuda --out runs/action_model --resume runs/action_model/last.pt
```

`--from-scratch` 不下载权重，主要用于环境验证，通常不适合少样本起步。RTX 5080 的 CUDA 运行速度、显存峰值仍需在实际机器测试。本机验证路径为 Mac CPU。

## 6. 独立评估

```bash
python -m ebo.cli evaluate --checkpoint runs/action_model/best.pt --device cuda --out runs/evaluation.json
```

报告包含每类 precision/recall/F1、混淆矩阵、低置信度比例、错误操作被判为正确操作的比例，并检查 train/val/test 数据泄漏。这是片段级评估，不能替代整段流程、时间边界或在线报警效果验收。

不要根据测试集反复调整模型或阈值。阈值在验证集上确定；用新的一批完整流程录像验收误报、漏报、延迟及稳定性。

## 7. RTSP 实时推理

```bash
python -m ebo.cli infer --checkpoint runs/action_model/best.pt --device cuda --log runs/live_session01.jsonl
```

PyQt 显示：

```bash
python -m ebo.cli monitor --checkpoint runs/action_model/best.pt --device cuda --log runs/live_session01.jsonl
```

离线录像回放使用 `--source /absolute/path/video.mp4`。命令行通过 Ctrl+C 结束；界面通过停止按钮结束。

推理与采集分离、缓冲有上限，读取 RTSP 使用 TCP 与超时，断线重连后清空旧窗口。低置信度、窗口不完整、断流或结果过期时输出 `unknown`，不会沿用旧的“正确”结果。模型时延、源时间、会话 epoch、候选动作、稳定动作和流程结果写入 JSONL。

摄像头内部缓存或网络传输延迟无法仅凭本机接收时间消除；`latency_seconds` 是预处理+模型计算时间，不是完整摄像头到屏幕延迟。首次结果需要积累一个窗口，再经过稳定化，多次确认会增加提醒延迟。

### 可选 SOP 顺序和时间

`sop.enabled` 默认 false。现场确认后启用，配置每步 action 与 min_seconds。

持续时间由稳定分类结果估计，未知/断线期间不累计，断线重连重新建立会话。错误动作、顺序错误或上一动作时间不足会产生提示。失败会话保持 failed，不能在后续步骤中自动变成合格；重新启动推理建立新会话。

当前没有人员进出触发/自动会话分割，不适合无人值守连续判多个员工。此版本不会直接控制门禁或自动放行。

## 文件结构

```text
config/project.yaml           类别、模型、训练、RTSP、流程参数
.env.example                  本地地址配置模板
ebo/video.py                  RTSP 接收、重连、时序窗口
ebo/data.py                   切片、检查、分组划分、泄漏检查
ebo/vlm.py                    本地大模型候选标签
ebo/model.py                  动作模型、统一预处理、checkpoint
ebo/training.py               微调、续训、评估
ebo/inference.py               连续推理、稳定化、日志
ebo/rules.py                  顺序和估计持续时间
ebo/ui.py                     PyQt 标注和监测界面
tests/                        自动验证
runs/                         运行后生成的验证结果、模型与日志（不随仓库发布）
```

## 已验证与尚未验证

可运行代码链路、预训练权重下载、训练/保存/加载、独立评估与本地视频推理由测试结果证明。真实工厂动作精度、摄像头画面与网络、RTX 5080 显存/速度、本地 Qwen 服务及长期运行稳定性，需要目标设备与现场数据验收；不能用合成视频的结果替代。

依据：[torchvision R3D-18 官方说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html)、[PyTorch 安装](https://pytorch.org/get-started/locally/)、[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)。

## 本机 RTSP 协议验证

可选安装 `python -m pip install -e '.[test]'`，从 [MediaMTX 官方发布](https://github.com/bluenviron/mediamtx/releases) 获取对应平台可执行文件：

```bash
python scripts/verify_rtsp.py --server /absolute/path/mediamtx --smoke-dir runs/smoke_new --out runs/rtsp_check_new
```

该脚本依赖前面的 smoke 验证结果，在 loopback 地址推送合成视频，实际使用 RTSP/TCP 采集，检查中断清空窗口、恢复新会话、录像与在线推理，最后关闭自己启动的服务和推流进程。默认端口18554，冲突可改 `--port`。

本机生成的协议验证结果保存在 `runs/rtsp_final/verification.json`；不是工厂 RTSP 摄像头验收。
