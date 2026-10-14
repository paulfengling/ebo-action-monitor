# 验证记录 · 2026-10-14

此页为开发者本机的历史验证记录。下列 `runs/` 路径是当时生成的本地产物，模型、录像、日志和截图不随公开仓库发布。读者可在自己的机器复现，不应将这些路径视为已附带的文件。

## 验证环境

Mac ARM64，Python 3.12.14，PyTorch 2.14.1，torchvision 0.29.1，OpenCV 5.0.0，PyQt6 6.11.0；无 CUDA。安装版本快照见 requirements-macos-verified.txt，不作为 NVIDIA CUDA 环境安装清单。

## 已运行的验证

| 项目 | 结果与依据 |
|---|---|
| 自动测试 | 17 项通过，覆盖稳定动作、流程、未确认标签、合成数据隔离、划分泄漏、视频连续性、停流、重连、VLM 候选不自动确认、标注保存、RGB/ROI/预处理 |
| 官方预训练权重 | 下载 R3D-18 Kinetics400 权重并加载，非随机权重冒充预训练 |
| 端到端链路 | runs/smoke_final/verification.json：切片清单、分类头微调、checkpoint、独立测试片段评估、本地视频流推理；4 次预测 |
| 全网络微调 | runs/smoke_full_finetune/verification.json：主干 stem 权重相对官方初始权重最大变化约0.0002008，确认主干更新 |
| 续训 | runs/smoke_resume/best.pt：成功恢复优化器、从第1轮续到第2轮；新输出目录可正常保存 best.pt |
| 命令行评估 | runs/smoke_final/cli_evaluation.json：train/val/test 分离检查通过并输出每类指标 |
| PyQt | 标注保存测试通过；runs/ui_check/annotation.png、monitor.png 为合成视频运行截图 |
| RTSP/TCP | runs/rtsp_final/verification.json：实际本机 MediaMTX 服务 + FFmpeg 推流，检查停流清空窗口、恢复后新 epoch、保存2个视频文件28帧、产生4次在线预测 |
| 本地 VLM 协议 | Mock 服务验证8帧请求和候选标签人工确认机制；没有实测真实 Qwen 服务 |

所有 smoke 模型均标记 demo_only，正式推理默认拒绝使用。合成视频的 accuracy 不作为现场效果；低置信度拒识与 unknown 输出是正常保护行为。

## 尚未验证

- 真实工厂动作类别、标签与测试集精度。
- 用户摄像头的 RTSP 地址、画面、密码、网络条件及摄像头内部延迟。
- NVIDIA RTX 5080 的 CUDA 安装、训练显存、网络计算速度及长期稳定性。
- 真实本地 Qwen-VL 服务的加载、显存和动作候选质量。
- 多人物/连续多员工会话、自动定位、物品计数或精确动作边界；当前项目为单人固定工位片段分类基线。

RTX 5080 部署机可执行 scripts/check_gpu.py 获取网络前向耗时与显存；完整摄像头到报警的延迟仍需现场单独测试。

## 复现方法

安装依赖后运行：

```bash
python -m unittest discover -s tests -v
python -m ebo.cli smoke --out runs/smoke_new --pretrained
python scripts/snapshot_ui.py --smoke-dir runs/smoke_new --out runs/ui_check_new
```

RTSP 协议测试还需安装测试依赖并下载对应平台的 MediaMTX，详见 README 的“本机 RTSP 协议验证”。合成数据和随机初始化的分类头可能使数值随环境变化；这里记录的是流程验证，不是精度承诺。
