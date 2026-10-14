# 本地视频流程拆分与逐帧预测

本工作流用 EBo 的 **Kinetics400 预训练 R3D-18 视频动作模型**测试本地 MP4。它读取视频像素，输出当前动作阶段，不依赖 AI-SOP 的 YOLO、手关键点或 LSTM。在线运行时不调用视觉大模型。

目前 `mirror_demo_01.mp4` 只有一段完整录像，因此提供 `fit-replay` 单视频拟合演示，用于验证“流程标签 → 模型 → 每帧结果导出”。**训练与回放来自同一视频，不是独立评估，不能作为新人员、新视频的准确率证据。**正常的 `train` / `split` 仍要求独立数据分组，不会通过随机拆分相邻窗口绕开检查。

## 镜子组装动作与片段

[config/mirror.yaml](../config/mirror.yaml) 定义六类，模型窗口为 2 秒、16 帧；图像缩放为 128×171 后中心裁剪为 112×112。候选流程结合用户提供的 `AI-SOP/full_timeline_10videos_template.csv` 与视频画面观察整理，边界经过粗略调整，尚未经工艺人员确认。准备脚本会将这些估计边界对齐原视频帧。

| 约略时间 | 类别 id | 动作阶段 |
|---|---|---|
| 0–2 秒 | `background` | 背景或未操作 |
| 2–34 秒 | `base_connect` | S1 底座与立杆连接 |
| 34–47 秒 | `mirror_align` | S2 镜面与支架对齐 |
| 47–55 秒 | `fasten_side_a` | S3 第一次侧边紧固，称侧 A |
| 55–69 秒 | `fasten_side_b` | S4 第二次侧边紧固，称侧 B |
| 69–91 秒 | `final_check` | S5 整体检查 |
| 91 秒–片尾 | `background` | 背景或镜头调整 |

S3/S4 表示**本视频先后操作的两侧**，不是已经建立工件坐标的左右侧定位；不能保证换一段视频后还会正确区分侧别。两侧能否交换顺序需要工艺确认。`sop.enabled` 默认关闭，步骤时间为占位值，不用动作分类结果直接判定装配合格。

原视频约 93.37 秒、2,800 帧，准备脚本生成 46 个互不重叠、至少 2 秒且不跨动作边界的片段，全部属于 `group_id=mirror_demo_01`：背景 2、S1 16、S2 6、S3 4、S4 7、S5 11。标注注明 `manual_observation`、粗略边界、单视频演示，仍需要工艺人员复核。

## 从视频准备到每帧导出

先按 README 安装项目依赖；如果需要叠字视频或 JPEG，还需安装 Pillow：

```bash
python -m pip install Pillow
```

以下命令在项目根目录执行。示例使用新的 `runs/mirror_example` 目录，避免覆盖现有 `runs/mirror_ebo` 产物。输入视频需自行放在项目目录，视频及模型不会随公开源码仓库发布。

```bash
# 1. 按人工观察流程切片，同时保存 timeline.csv、segments.csv 和 reviewed.jsonl。
python scripts/prepare_mirror.py \
  --video mirror_demo_01.mp4 \
  --out runs/mirror_example/input

# 2. 检查训练片段、标签、长度和重复文件。
python -m ebo.cli --config runs/mirror_example/input/config.yaml validate \
  --manifest runs/mirror_example/input/reviewed.jsonl

# 3. 提取预训练特征，拟合本视频的分类头。
python -m ebo.cli --config runs/mirror_example/input/config.yaml fit-replay \
  --manifest runs/mirror_example/input/reviewed.jsonl \
  --out runs/mirror_example/model \
  --device cuda

# 4. 每个源帧保存一条记录和叠字 JPEG，同时生成完整叠字视频。
python -m ebo.cli --config runs/mirror_example/input/config.yaml predict-video \
  --checkpoint runs/mirror_example/model/replay_model.pt \
  --video mirror_demo_01.mp4 \
  --out runs/mirror_example/predictions \
  --device cuda \
  --stride 1 \
  --save-frames \
  --reference-timeline runs/mirror_example/input/timeline.csv \
  --allow-demo-model
```

没有 NVIDIA GPU 时可将 `--device cuda` 改为 `cpu`；支持 Metal 的 Mac 可用 `--device mps`，本次实际完整导出使用此方式。耗时需要在实际设备测量。首次拟合会下载 torchvision 官方预训练权重，需要网络或提前缓存；已有权重和 checkpoint 后，本地推理不需要云端服务。

`fit-replay` 固定 R3D-18 主干，按每个片段提取一个中心窗口的 512 维特征，只训练 `fc` 线性分类头。特征标准化合并回分类头后，checkpoint 仍使用普通 R3D-18 推理路径。输出 `replay_model.pt`、`features.npz`、`history.json` 和 `calibration.json`；其中的 `training_fit_accuracy` 仅说明训练片段拟合程度，不能当成独立测试准确率。

同视频模型标记 `demo_only=true`、`calibration_only=true`、`evidence_scope=same_video_replay`，加载时必须显式使用 `--allow-demo-model`。视频画面与每帧结果都会保留该证据范围。

## 每帧记录是什么意思

`predict-video` 按源视频顺序解码。每一帧使用该时刻之前的 2 秒历史采样 16 帧，不读取未来帧，也不把叠加后的画面喂给模型。

本视频前 **60 帧（编号 0–59）**历史不足完整窗口，记录 `warmup=true`，动作及候选为 `unknown`，概率为 `null`，不会伪造预测。从第 60 帧开始，`--stride 1` 对每帧独立执行模型，预计产生 2,740 次模型推理和总计 2,800 条记录。

| 字段 | 含义 |
|---|---|
| `frame_index`、`timestamp_seconds` | 从 0 开始的源帧编号；视频时间为帧编号除以容器 FPS |
| `candidate`、`candidate_name` | 本次模型概率最高的原始候选动作，低置信度时也保留候选 |
| `confidence`、`probabilities` | 模型 softmax 的候选分数及全部六类分数；不代表质量合格概率。CSV 展开为 `probability_*` 列 |
| `action`、`action_name` | 经过置信度和连续确认后的稳定动作，不满足条件时为 `unknown` |
| `decision` | 配置的类别属性 `kind`；本次 `correct` 只表示正常动作类别，不表示装配质量合格 |
| `stable_confirmation_count`、`stability_updated` | 当前确认次数、本帧是否更新稳定判定 |
| `sampled_frame_indices`、`window_*` | 模型实际使用的历史帧编号与时间区间 |
| `prediction_frame_index`、`prediction_reused` | 产生预测的源帧、当前记录是否复用旧预测 |
| `warmup`、`latency_seconds` | 是否预热、本次模型计算耗时；不等于摄像头端到端延迟 |
| `reference_label` | 可选人工时间线参考标签，**仅用于导出对照** |
| `sop` | 规则引擎状态；本配置默认关闭 |
| `evidence_scope`、`demo_only`、`calibration_only` | 训练来源与演示范围 |

默认置信度阈值为 0.70，稳定动作需要连续 3 次确认。确认按源视频时间每 0.5 秒进行，第一次确认后还需至少约 1 秒，不能因为每帧运行模型就瞬间完成确认。候选改变或置信度不足会立即撤销旧稳定动作。因此候选与稳定动作可以不同；即便 `confidence` 很高，稳定确认未完成时 `action` 仍然是 `unknown`。模型输入是一个 2 秒视频窗口，不是仅凭当前单张图片分类。

`--reference-timeline` 只在模型、稳定化及 SOP 运算完成后附加人工标签，既不喂给模型，也不触发步骤推进。它不是模型预测，不应把人工参考时间直接当成识别结果。

如果将 `--stride` 改为大于 1，仅每隔指定帧数执行模型；中间帧仍保存，但明确标记 `prediction_reused=true`，不推进稳定确认或 SOP。本次每帧测试应保持 `--stride 1`。

离线时间使用 `frame_index / container_fps` 的近似 CFR 时钟，不是原始逐帧 PTS。本片核对两者的最大差异约为 18 毫秒。可变帧率视频需要先转为固定帧率，才能按这套时间基准对齐。因果窗口包含旧阶段画面，步骤切换时存在窗口和稳定化延迟；粗略人工边界也会影响对照。

## 导出文件

```text
runs/mirror_example/
  input/
    config.yaml          本次配置
    timeline.csv         七段粗略人工流程及帧对齐时间
    segments.csv         46 段切片范围
    reviewed.jsonl       单来源训练清单
    preparation.json     数据来源、片段统计与限制
    clips/*.mp4          训练片段
  model/
    replay_model.pt      单视频演示模型
    features.npz         冻结主干的训练特征
    history.json         分类头拟合历史
    calibration.json     来源与训练拟合统计
  predictions/
    predictions.csv      一帧一行，可用 Excel 查看全部类别概率
    predictions.jsonl    一帧一条，包含嵌套概率、采样帧和 SOP
    annotated.mp4        保留源帧率的完整叠字视频，不含原音频
    frames/*.jpg         每个源帧的叠字画面，编号与 CSV 对应
    summary.json         来源哈希、参数、总帧数、耗时和完成状态
```

本机实际产物位于 `runs/mirror_ebo/input/`、`runs/mirror_ebo/model/`、`runs/mirror_ebo/predictions/`。输出目录不会自动覆盖；重新运行请选择新目录。检查 `summary.json` 的 `status` 和 `frames_exported`，区分完整导出、调试截断、解码不完整或失败。

本次已完整运行并重新解码检查：2,800 条 CSV/JSONL、2,800 张可解码 JPEG、叠字视频 2,800 帧；前 60 帧预热，其余 2,740 帧均执行新预测，复用次数为 0。27 项自动测试通过。`verification.json` 保存核对结果；另整理了 `preview.jpg` 和 `action_segments.csv`，后者按实际稳定模型输出合并连续帧，保留 `unknown` 区间，不使用参考标签生成阶段。全部产物仍属于同视频训练回放。

中文叠字需要 CJK 字体。找不到系统字体时用 `--overlay-font /path/to/font.ttf` 指定；只要 CSV/JSONL 时去掉 `--save-frames` 并加 `--no-video`。调试可加 `--max-frames 90`，但这是部分视频导出。加 `--save-raw-frames` 可将未叠字的 JPEG 另存至 `raw_frames/`；Python API 对应 `ebo.offline.run(..., save_raw_frames=True)`。

## 换成真实独立测试

准备新的人员、日期或完整独立录像，按相同动作定义标注，保留独立 `group_id`。使用项目原有 `split → train → evaluate`，再将独立视频传入 `predict-video`；人工参考仍只作对照。重新确认动作定义、侧别、边界和阈值，不能把单视频拟合模型直接视为现场成品模型。

当前输出是动作阶段分类，**没有部件检测框、胶带张数、螺丝实例计数、扭矩测量或装配质量判定**。识别到“紧固”只能说明画面类似该动作，不能证明已拧紧；S5 也只是“整体检查”动作，不是成品验收通过。
