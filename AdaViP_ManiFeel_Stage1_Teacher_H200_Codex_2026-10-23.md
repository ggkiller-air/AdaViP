# AdaViP — ManiFeel 新版 Stage 1 Teacher：分支整理、重构与 H200 训练指令（2026-10-23）

> **执行位置**：H200 训练服务器  
> **指定开发分支**：`feature/1023stage1_2`  
> **范围**：保护旧实验、整理新版工作区、复用旧基础设施、实现新版 Stage 1 Teacher、检查与正式训练（最多 400 Epoch）  
> **当前任务授权**：完成必要测试后直接启动训练；不得自动生成 Utility Labels、启动 Stage 2 或在 H200 执行 ManiFeel Rollout。

请先阅读已提供的 `AdaViP_Session_Handoff_2026-10-22.md`，参考其中 ManiFeel 数据、Diffusion Policy、Normalizer 和评估接口；但**以本文的新架构为准**，不再沿用旧 `C0 + C1 + Adapter` 结构。先核对实际代码与数据契约，不要猜测服务器路径或已有接口。

## 0. Git 分支、复用与工作区整理（**务必先做**）

**请先检查当前 Git 仓库的 HEAD/分支、`git status --short`、未提交改动、未跟踪文件，以及现有 Stage 0/1/2 文件结构。不能直接在旧分支上覆盖重构。**

1. 在保护原有工作的前提下，切换到指定分支 **`feature/1023stage1_2`**：若不存在则从当前有效基线创建，若已存在则检查内容并切换；禁止 `reset --hard`、强制覆盖分支或丢弃未提交修改。若工作区有改动，先用可恢复方式妥善保存并记录（例如有说明的备份提交/安全 stash）；对未知改动不盲目迁移或删除。
2. **先做旧代码清点表**：哪些属于可复用基础设施（ManiFeel 数据/Zarr 读取、动作与 Episode 索引、Normalizer、Diffusion U-Net、EMA/Checkpoint、评估兼容接口、日志、测试），哪些属于已废弃的旧 C0/C1/Adapter、双帧输入、9D Utility/Stage 1 训练逻辑。实际检查依赖后再复用，避免复制错误假设。
3. **在新分支进行干净的架构重构**：用符合现有仓库风格的独立 `stage1_v2` 模型、配置、训练入口和测试组织新版代码；复用可信公共模块，不要重新造已有 Diffusion/数据轮子。允许在**新分支内**移除或归档已确认不再使用的旧版专属源码/入口，前提是不会破坏公共依赖，并且在 Git 旧分支或提交中可追溯、可恢复。不要为了整理而大范围搬迁整个仓库。
4. **旧实验资产必须保护**：不要删除或覆盖旧 `outputs/`、Checkpoints、Utility Labels、数据、文档、旧实验日志及对应 Git 历史；这些大文件可能被 Git 忽略，切分支**不代表**有备份。新版输出目录使用独立版本化命名（如 `outputs/manifeel_stage1_v2_*`），避免旧/新 Checkpoint 混用；旧版历史报告可留在 docs 的 legacy/历史命名下，但不强制迁移。
5. **清理的边界**：仅整理新版源码命名、冗余导入和失效的新版临时文件；清理前列出将变更/移除的文件。不得运行 `git clean -fdx`、递归删除旧目录、清空输出目录，或删除尚未确认用途的文件。
6. 在启动训练前执行 `git status`，记录当前分支、Commit、关键文件路径、复用与替换清单；建议完成可复现的新版 Coding Commit（不纳入训练权重/数据）。确保新训练入口不加载旧 C0/Adapter 或旧 9D 标签。

**目标：最后新分支中的活跃 Stage 1 路径清楚、目录整洁；旧分支和旧训练产物完整保留。** 完成该工程整理后，继续执行以下模型实现及训练指令；不需要另行等待许可，但如发现无法安全隔离、数据契约严重不符等阻塞问题，应停止危险操作并说明。


## 1. 数据与传感器编码

使用 ManiFeel 全部 **9 任务、400 条 Demonstrations**。

- **Observation History：6 帧**，以当前时刻 `t` 为锚点，观测 `[t-5,...,t]`。
- **视觉/触觉**：Front、Side、Wrist RGB + Left、Right GelSight，共 5 个独立 ResNet18；每帧输出 512D 特征。
- 各路 6 帧特征加入 Temporal Embedding，再通过**轻量 Temporal Transformer** 进行时间建模，压缩为该传感器的一个 **256D Token**。
- **Proprioception**：6 帧 7D EEF Pose（3D position + xyzw quaternion，不是 joint qpos），通过 MLP + Temporal Encoder 压缩为一个 256D Token；不要对 EEF Pose 使用 ResNet。
- **Language**：任务固定 CLIP 512D Embedding，投影为一个 256D Token；没有语言泛化的实验假设。
- 最终得到 **7 个 Sensor Tokens，形状 `[B,7,256]`**（5 路图像、1 路本体、1 路语言）。

重新建立满足完整 6 帧历史、Action Chunk 与 Future Targets 的有效窗口索引；严格检查 Episode Boundary，**不要复用旧双帧 65,265 窗口的数量与索引**。

## 2. State Tokenizer 与 Action Backbone

借鉴 MS-Bot：

1. 使用**最新一帧**的 5 路视觉/触觉特征、当前 Proprioception、Language Embedding。
2. 最近 **8 步已执行 Action History** 输入 `LSTM(hidden=128)`；History 必须是推理时可获得的过去动作，不得混入预测窗口中的未来动作。
3. 将上述信息 Concatenate，经 MLP 得到 `z_t [B,256]`。
4. Cross-Attention 使用 `Q=z_t`、`K/V=7 Sensor Tokens`，输出 256D Conditioning，送入一个 Conditional Diffusion U-Net 预测动作。

保持原 ManiFeel 动作契约：

- Action prediction horizon `H=16`，执行步长 `T_a=8`。
- Diffusion training timesteps = `100`。
- Action 7D；部分任务为 6 个有效维度，继续正确使用 Action Loss Mask。
- 旧数据契约中 Chunk 第 0 位为 `a_(t-1)`，第 1 位才是 `a_t`；实现新接口时必须检查并维持索引一致性。

## 3. Future Vision / Tactile Prediction

从当前 `z_t` 与示教动作条件预测**未来 4 帧**视觉与触觉 Latent，而不是 RGB 像素：

- Future Targets：`t+1,...,t+4` 的 3 路视觉与 2 路触觉特征，由对应的 **EMA ResNet Encoder** 编码，`stop-gradient`。
- Future Predictor 可以接收 `a_t,...,a_(t+3)`；根据上述旧 Action Chunk 契约，检查使用索引 `[1:5]`。
- 未来真实观测和未来示教动作**只允许进入辅助 Future 分支**，不得进入 State Tokenizer、Cross-Attention 或 Diffusion Action Conditioning。
- EMA Target Encoder 从本次 Stage 1 Online Encoder 初始化，后续仅通过 EMA 更新，不反向传播。
- Vision Future Loss 与 Tactile Future Loss **分别记录**；跟踪 Latent 方差与退化，不能把低 MSE 自动解释为学会动力学。

训练目标：

\[
\mathcal L=\mathcal L_{\mathrm{action}}+\lambda_V\mathcal L_{\mathrm{future\ vision}}+\lambda_T\mathcal L_{\mathrm{future\ tactile}}.
\]

启动正式训练前进行梯度量级与方向检查，选择有依据的 `λV/λT`；**不要直接沿用旧 Stage 1 Joint Training 的极小辅助系数**。记录选取逻辑。

## 4. Modality Dropout

沿用 Stage 0 原有比例：**80% Full + 20% Masked**，Masked 时从以下 8 种移除条件等概率采样：

1. Front RGB
2. Side RGB
3. Wrist RGB
4. Left Tactile
5. Right Tactile
6. Proprioception
7. All Vision
8. All Tactile

**Language 永不 Dropout。**

同一路被 Mask 时，必须同时屏蔽它进入 State Tokenizer 与 Cross-Attention K/V 的路径，防止泄漏。六帧 Sensor History 作为整体移除，不是随机移除其中单帧。采用与现有实现兼容、可学习且在训练时受过覆盖的 Missing Token / Presence 机制，并记录具体实现。

这 8 种 Mask 将用于**后续独立的 8D Counterfactual Utility GT 标注**；本次训练阶段不生成标签，也没有 Utility Head。

## 5. 训练设置与检查

**Stage 1 Teacher 的模型权重从头初始化**，不继承旧双帧 C0 Teacher 的参数；**这不妨碍复用旧版可靠的数据与训练代码**。本次不使用 HyperNet、Utility Head 或 Adapter。

- 联合训练：5 路 ResNet、Temporal Encoders、Proprio/Language Projection、Action History LSTM、State Tokenizer、Cross-Attention、Future Predictor、Diffusion U-Net。
- 使用全部新的有效 Demonstration 窗口，按任务均衡策略采样；无放回遍历、耗尽后重洗牌。明确记录每 Epoch 定义、每任务窗口数和覆盖情况。
- 在 H200 检查 Batch `8/16/32` 的显存峰值与吞吐后选择适当 Batch Size；记录 LR、Optimizer、Scheduler、EMA Decay、随机种子及输入归一化策略。
- 训练前完成：Shape/Gradient、Mask 泄漏、Normalizer 恢复、Action/Future Alignment、跨 Episode 边界、Diffusion 推理、短步数数值稳定性等必要测试。
- 随后**直接启动正式训练，最多 400 Epoch**，不必再次征求许可。
- 持续记录 Action/Future Loss、各任务覆盖、梯度诊断、EMA Latent 方差及 NaN/Inf 情况。数值错误或数据泄漏应及时停止并报告。

## 6. Checkpoint 保存

**必须保留以下 Epoch 的独立完整 Checkpoint：**

`1, 2, 5, 10, 20, 50, 100, 200, 300, 400`

每份包括模型（含 EMA Target Encoder）、Optimizer、Scheduler、Normalizer、RNG、采样状态、全部配置和数据/模型版本信息，确保精确恢复。**不要覆盖 Epoch 1/2**，它们将作为 Stage 2 早期初始化的候选。

## 7. 执行完成后的交付

请提供：

1. 当前 Git 分支 `feature/1023stage1_2`、Commit、旧版资产保护情况、复用/移除的源码清单；关键模块结构、参数量。
2. 训练前检查结果、所选 Batch/LR/Future Loss 权重及理由。
3. H200 作业编号、输出路径、日志路径。
4. 各指定 Epoch 的训练统计与可恢复 Checkpoint 清单；如尚未完成 400 Epoch，则如实报告当前状态。

**明确边界：此次仅 Coding + Stage 1 Teacher 训练。不要自动运行 RTX/4090 闭环 Rollout，不要生成 Utility Labels，不要启动 Stage 2。** Teacher 选择将在独立的 RTX/4090 ManiFeel 闭环评估后决定。
