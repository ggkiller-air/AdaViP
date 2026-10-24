# AdaViP 研究交接：Stage 0 → Utility Labels → Stage 1（截至 2026-10-22）

> **用途**：将本文件直接上传到新的 ChatGPT Session，继续讨论 ManiFeel Stage 1 的 Adapter Adaptation、联合训练与 Stage 2 设计。本文合并了本轮对话、H200 Codex 的已上传 `docs/` 和先前交接记录。它是一份**已执行实验、人工选择与待验证假设**的分层记录，不能把候选架构或训练损失改善写成论文结论。
>
> **最重要的当前状态**：ManiFeel Stage 0 已完成；九任务 Teacher **固定为 Epoch 500 EMA**。**65,265 个窗口、9D、加权 K64 Utility Labels 已全部生成并通过完整性检查**。Stage 1 的 **C1 + Attention Pooling + Residual Adapter + Utility/Future Heads 已实现**，包括 Normalizer 和完整 Diffusion 推理的零误差初始化核验。独立正式训练的 **Stage 1 Phase 1（Representation Warmup）已在 H200 完成 10,000 steps**；人工建议使用 **Warmup Step 9000** 作为 Phase 2 起点。**Phase 2 Adapter Adaptation、Phase 3 U-Net Joint Fine-tuning、Stage 1 闭环 Rollout 和 Stage 2 均未开始**。
>
> **用户的协作偏好**：讨论简洁、严谨、承认不确定性；不要把一般技术问题每次都套回 AdaViP。生成 Codex 指令时不必写冗长限制，但必须指明服务器、模型/数据版本、样本/步数、训练或评估范围，并且不要在用户未授权的情况下自动进入下一阶段。

---

## 0. 信息来源及优先级

1. **当前阶段的直接依据**：`docs/stage1_representation_warmup_results_20261022.md`、`docs/stage1_representation_warmup.md`、`docs/stage1_core_architecture.md`、`docs/stage1_training_readiness.md`、`docs/stage1_controlled_training_results_20261022.md`。
2. **标签与 Teacher 的直接依据**：`docs/stage1_utility_full_labels.md`、`docs/stage1_utility_label_pilot.md`、`docs/stage1_utility_k64_precision.md`、`docs/stage1_label_preparation.md`、`docs/full_sensor_stage0.md`、`docs/stage0_utility_k8_diagnostic.md`、`docs/stage0_offline_utility_diagnostic.md`。
3. 早期方案：`AdaViP_Session_Handoff_2026-10-20(1).md` 和 `AdaViP_Full_Framework_Codex_Audit_2026-10-17(1).md`；含部分**已被实际实现取代**的规划，不得覆盖最新记录。
4. 本交接记录根据用户提供的 H200/Codex 报告和会话信息整理；**并未远程登录服务器读取实时工作树、Slurm 或全部训练数组**。新 Session 如需改变代码，应先让服务器端 Codex 检查实际 HEAD/源码、模型配置及最新任务状态。

**服务器分工**：H200 训练服务器负责冻结 Teacher 的离线 loss/utility、Stage 1/2 训练和数据探针；RTX/4090 仿真评估服务器负责 ManiFeel/UniVTAC Rollout（包括 Masked Rollout）。不要臆造新的主机名或把 H200 离线 Loss 当作仿真成功率。

---

## 1. 研究目标与三阶段框架

**AdaViP**：Utility-Supervised Adaptive Visuotactile Perception for Robotic Manipulation。核心问题是能否通过**动作相关的显式模态 Utility 监督**，学习出包含交互历史和接触动态的表示，并最终优于仅用 Action Loss 的融合策略。

- **Stage 0 — Reference Teacher（已完成）**：普通 full-sensor multi-task Diffusion Policy，7 路传感器、随机 pre-fusion mask，只有 action epsilon-denoising loss；无 C1/Utility/Future/HyperNet。
- **Utility Label（已完成）**：冻结 Epoch 500 EMA，在每个有效示教窗口比较 Full 和九种 masked observation 下的**同噪声、同 timestep 的去噪 MSE**，离线存 9D signed utility。其含义是 Teacher 在 demonstration action 上的 denoising-loss sensitivity，**不是**真实因果模态贡献、Shapley value 或 closed-loop SR 差。
- **Stage 1 — Interaction Representation（架构已实现，Phase 1 已完成）**：冻结 Stage 0 图像编码和旧融合 `C0`，另建 C1 时序 Transformer，利用 Utility、Future Tactile、Action 三个监督学习 `z_t` 并逐步让新表示适配旧 U-Net。
- **Stage 2 — Utility-Guided Adaptive Fusion（未 Coding/训练）**：初始计划固定或谨慎微调 Stage 1 的 Encoder/C1/Utility Head；用预测的 signed utility 和 `z_t` 驱动 Query/Attention Bias、FiLM/HyperNet 变换，再由新的 C2 Cross-Attention 选择动态模态特征；保留 residual identity 保护 Stage 1 DP。具体 Q/K/V 机制、训练方式和性能尚未确定。

关键研究边界：`U(z)` 是 `z` 的函数，**并不增加新的输入信息**。研究贡献要靠 utility-supervised inductive bias 和匹配容量的对照与闭环实验成立；不能只因为 Utility Head 可拟合或 Future Loss 下降就宣称自适应感知有效。

---

## 2. ManiFeel 数据、Stage 0 实际结构及 Teacher

### 2.1 数据与动作契约

- **9 个 ManiFeel 任务，400 条 Demonstration 轨迹、71,265 个原始帧**：Peg Insertion、USB Insertion、Power Plug Insertion、Gear Assembly、Nut/Bolt Assembly、Bulb Installation、Peg Reorientation、Object Search、Ball Sorting。
- **7 路策略输入**，顺序为 `[front, side, wrist, left_tactile, right_tactile, state, language]`。3 路 RGB `[B,2,3,256,256]`；2 路 TacRGB `[B,2,3,320,240]`；State `[B,2,7]` 为 EEF 3D position + xyzw quaternion（**不是** joint qpos）；Language 为每任务固定 512D CLIP embedding（**不能宣称语言指令泛化**）。
- Observation history `T_o=2`、Action prediction horizon `H=16`、执行步长 `T_a=8`；Action backbone 7D，部分任务只有 6 个有效维度，使用 loss mask。
- 以机器人观测 `t` 为 anchor，观察 `[t-1,t]`，**Action Chunk 第 0 位为 `a_(t-1)`，第 1 位才是 `a_t`**；此对齐会直接影响 Stage 1 Future Predictor。
- 每个完整有效窗口必须有两帧真实历史和 16 步真实 Action，不使用 episode 边界 padding；全量离线标注共有 **65,265** 窗口（400 条轨迹）。本地演示没有可靠逐帧 timestamps，不能自行把 step offset 换算为秒。

### 2.2 Stage 0 网络

```text
3 RGB + 2 TacRGB (各 2 帧) ─→ 五个独立 ResNet18(GroupNorm)
                                      每帧 512D
  每路 [512D,512D] concat → Linear(1024→256) + LN + SiLU
State 双帧 concat → 256D；Language 512→256D
       ↓
7 sensor tokens [B,7,256]
       ↓  每路 mask 时以 learned missing token 代替，并附 7 presence flags
C0: Concat(7×256 + 7 flags = 1799) → MLP → c0 [B,256]
       ↓
Conditional 1D U-Net (epsilon prediction, 100 diffusion training timesteps)
       ↓
Action denoising loss
```

Mask 共有 **Full + 9 removals**：依次 V1/V2/V3、T1/T2、P/L、all-vision、all-tactile。Stage 0 默认 **80% Full + 20% 九种 removal 等概率**；在首次跨传感器融合前 mask。绝不把 `mask vision` 等同于 patch-level attribution。

训练最初使用完整轨迹 train/val split；到 **Epoch 300（324,600 optimizer updates）** 后，用户明确改为把原 validation demonstrations 纳入 Stage 0 续训，冻结原 Normalizer 而不是重拟合。因此 Epoch 300 后的 Validation Loss **不再是独立的泛化指标**，后续主要用 RTX 闭环 Rollout 选教师。

Stage 0 实际总训练到 **Epoch 600 / global step 658800**；用户最终选 **Epoch 500 EMA / global step 547400** 作为所有九任务的统一 Teacher，不按任务拼接 Epoch。

**Teacher 身份（必须复核）**：

- Release：`/data/wangzihao/outputs/manifeel/checkpoint_archive/full_sensor_20261019_epochs300_400_500/package/checkpoints/epoch500.ckpt`
- SHA256：`7936f6ac9084d2afaca6bf35d0322dd3ad9c8339da71f87e8640d59f4c86c3eb`
- Stage 0 Normalizer SHA256：`2d79d95359de2b8f9b68f87850a3902bbecdeea3c4de8d4acb533c0ea7db7571`
- 必须加载 `payload['state_dicts']['ema_model']`、匹配 checkpoint 中的配置和动态创建的 Normalizer；**不混用、不重拟合、不静默跳过权重**。

### 2.3 Stage 0 RTX/4090 正式 Full-input Rollout

每任务 30 环境初态、每次最多 500 env steps，配对主 seed `100000`、Diffusion RNG seed `42`；指标 SR 为 episode 内**曾经**触发 `_check_success()`，官方 `mean_score` 是轨迹中最大 wrapper reward 均值，两者定义不同（该轮恰巧数值相同）。

| 任务 | E300 EMA | E500 EMA | E600 EMA |
|---|---:|---:|---:|
| Peg Insertion | 5/30 | 6/30 | 6/30 |
| USB Insertion | 5/30 | 5/30 | 3/30 |
| Power Plug | 8/30 | 7/30 | 7/30 |
| Gear Assembly | 7/30 | 6/30 | 7/30 |
| Nut/Bolt | 14/30 | 12/30 | 16/30 |
| Bulb Installation | 18/30 | 21/30 | 19/30 |
| Peg Reorientation | 14/30 | 18/30 | 16/30 |
| Object Search | 1/30 | 1/30 | 1/30 |
| Ball Sorting | 10/30 | 14/30 | 16/30 |
| **总计 / Macro SR** | **82/270 = 30.37%** | **90/270 = 33.33%** | **91/270 = 33.70%** |

**E600 vs E500 只有 +1/270 成功，不显著**，且 E500 已完成深入 Utility 诊断，所以统一选 E500。E500/E600 Object Search 均为 1/30，且是短暂满足成功判据，**不能把该任务写成 Teacher 已掌握搜索**。Bulb 等也存在曾成功与末步仍成功的差异。

---

## 3. Utility 标签是什么，为什么从 K8 改为加权 K64

### 3.1 标签定义和计算

对于一个 Demo 窗口 `o_t` 和真实 clean Action Chunk `a_demo`：随机抽 diffusion training timestep `k ∈ {0,…,99}` 与 Gaussian `ε`，构造 `x_k = sqrt(alpha_bar_k)·a_demo + sqrt(1-alpha_bar_k)·ε`，冻结 Teacher 各评估一次 Full 与 9 Mask 的 **epsilon prediction MSE**。同一个 draw 的十条件严格共享 `x_k`、`k`、`ε`、target、Normalizer、6D/7D loss mask。

\[
 u_t^{(m)}=\mathbb E_{k\sim\mathrm{Unif}(0,\ldots,99),\epsilon}
 \big[\mathcal L_{\mathrm{mask}(m)}-\mathcal L_{\mathrm{full}}\big].
\]

顺序固定 `[V1/front,V2/side,V3/wrist,T1/left,T2/right,P/state,L/language,vision_group,tactile_group]`。每个机器人窗口最终**只得到一组 9D label**；`K` 是对该窗口 Monte Carlo 抽取 `(k,ε)` 的次数。**不是**从纯 Gaussian Noise 执行 100 次完整反向采样，也不是生成 64 组最终标签。

为什么不直接完整采样 Action 后比较 Demo MSE：多模态 Diffusion Policy 可输出多个合理动作；和单条 Demo 不一致未必代表策略错误，且完整反向采样成本高。当前 Noise-Loss Utility 与 Teacher 训练目标一致，但并不保证与 closed-loop 成功率一致。

### 3.2 标签协议的关键决策时间线

1. **最初 720 窗口固定四档 K8**：9 任务 × 5 轨迹 × 16 时间点；`k={12,37,62,87}` 各两个独立 Noise。该估计是**四个固定 timestep 的平均**，并非均匀 100 timestep 期望。
2. **作业 377460：固定 K8 vs 四组随机分层 K8**。同窗同 E500 下九维在九任务均呈随机 K8 均值增加：Vision/group **0.00773863→0.01117143（+44.36%）**，Left **0.00035242→0.00078280（+122.12%）**，Right **0.00063275→0.00124089（+96.11%）**，Tactile/group **0.00094299→0.00193923（+105.65%）**。固定 K8 遗漏了 `k=0–11` 的强响应；该区间贡献该次抽样 Tactile signed draw-sum 的约 **78.72%**。不能说随机估计器“有偏”而固定 K8 正确；两者目标不同。
3. **随机 K8 单窗触觉噪声仍过大**：Left/Right/group 的采样噪声 RMS / 去噪信号 RMS 为 **1.054/0.885/0.923**，近零和符号不稳定明显；故**不批准 K8 全量标注**。
4. **作业 377641：加密分层 K64 精度诊断**。对原 720 窗口做两次**独立 K64**，并拆分得到独立 K16/K32；按区间长度加权，仍估计均匀 100 步目标。全 81 个任务×维度启发精度门限：K16 11/81、K32 34/81、K64 54/81；K64 优于低 K，但**不等于九维全部精确收敛**。
5. **作业 377780：正式 2,880 窗口 Pilot**，固定 9×10 trajectories×32 timepoints，K64 正式生成。左/右触觉并无数值崩溃或整维塌缩；监督分布支持开展 Student 试验。采用 pooled per-dimension standardization + Huber(delta=1) 为优先候选、保留 MSE 对照；signed/near-zero/negative 全保留。
6. **全量标签**：用户授权 65,265 完整窗口；分块、断点续算、双 H200 并行经历 `flock ENOLCK` 和管理员 GPU 使用规则等工程问题后，最终产物已经被 Stage 1 readiness / controlled / warmup **逐文件 SHA256 与 255 shard 审计并实际读取使用**。早期 `stage1_utility_full_labels.md` 中的 PENDING/未完成文字是历史快照，**不代表当前未完成**。

**正式 K64 算法**：

| diffusion k 区间（闭区间） | 0–11 | 12–24 | 25–49 | 50–74 | 75–99 |
|---|---:|---:|---:|---:|---:|
| 每窗采样数 | 16 | 12 | 12 | 12 | 12 |
| 区间总概率权重 | 0.12 | 0.13 | 0.25 | 0.25 | 0.25 |

每区间内独立均匀有放回抽 `k`、独立 Noise，**先算区间内均值，再按 0.12/0.13/0.25/0.25/0.25 加权**，不能直接平均全部 64 draw（那会改变均匀 100 步目标）。同 draw 的 Mask 共用随机性。

### 3.3 K64 关键精度数字（720 窗口）

| 模态 | 随机 K8 噪声/信号 RMS | 加密 K16 | K32 | K64 | K64 两独立重复 Pearson |
|---|---:|---:|---:|---:|---:|
| Front | 0.792 | 0.312 | 0.222 | 0.158 | 0.952 |
| Side | 0.758 | 0.301 | 0.213 | 0.151 | 0.944 |
| Wrist | 0.672 | 0.279 | 0.198 | 0.140 | 0.938 |
| Left Tactile | 1.054 | 0.424 | 0.300 | **0.212** | 0.932 |
| Right Tactile | 0.885 | 0.382 | 0.269 | **0.191** | 0.933 |
| State | 0.763 | 0.302 | 0.214 | 0.152 | 0.914 |
| Language | 1.081 | 0.428 | 0.303 | 0.214 | 0.880 |
| Vision Group | 0.553 | 0.226 | 0.161 | 0.114 | 0.956 |
| Tactile Group | 0.923 | 0.390 | 0.277 | **0.197** | 0.910 |

Group Tactile 独立重复相关随 K16/K32/K64 为 **0.712/0.831/0.910**。K64 Tactile 左/右/group 的采样方差约 **97%** 仍来自 `k=0–11`，表明低 diffusion k 对 noise-loss 差异高度敏感；**小 k 是反向去噪末期**，而非机器人时间的接触后期，不能混淆。

K64 的低信号窗口仍有诊断误差带与近零区相交，不要求这些窗口的正负号能精确恢复；提高 K 只能降低随机估计误差，不能修复 Teacher 自身的偏差。

### 3.4 正式 Pilot 的九维标签分布（2,880 窗口）

| Utility | 均值（raw loss 单位） | 标准差 | `abs(u)≤1e-4` |
|---|---:|---:|---:|
| Front | 0.003434 | 0.003338 | 6.32% |
| Side | 0.004528 | 0.003892 | 3.58% |
| Wrist | 0.005070 | 0.003461 | 0.03% |
| Left Tactile | **0.000932** | 0.001188 | **14.34%** |
| Right Tactile | **0.001313** | 0.001571 | **12.26%** |
| State | 0.007495 | 0.004606 | 0.00% |
| Language | 0.001562 | 0.001185 | 1.39% |
| Vision Group | 0.011148 | 0.007224 | 0.00% |
| Tactile Group | 0.002101 | 0.001873 | 4.03% |

左/右两个独立 K32 半份的相关为 **0.854/0.884**；K64 噪声/信号 RMS **0.216/0.196**。Object Search、Ball Sorting 触觉近零比例高，但轨迹后段的 Tactile Utility 又明显上升；这些只是演示序列位置的描述性动态，不可直接断言机器人“接触阶段”的因果贡献。

### 3.5 全量标签产物与工程接口（已完成）

- 项目内存储：`outputs/stage1_utility_labels_ep500_k64/`，而不是 `/data/wangzihao/outputs/`；已从 Git 排除 `outputs/`。
- `COMPLETE.json` 与 255 个 `shards/*.npz` 的 hash/重建检查已在后续 Stage 1 preflight 实际通过。
- `window_index.npy`：`[65265]` structured index（task/episode/timestep/source_index 等）；`utility.npy`：**float64 `[65265,9]`**；`utility_variance.npy`：**float64 `[65265,9]`**（估计器 MC 方差，不是跨窗 SD）；`weighted_losses.npy`：**float64 `[65265,10]`**；`manifest.json`、`metadata/teacher_normalizer.npz` 和原始分块完整保存。
- 已验证所有 **65,265 个训练样本**与 source Zarr 的 task/episode/timestep/action/future 索引一一对应；不通过简单按数组位置猜测 Row 对应。
- 全量一次 K64 共 `65265×64×10=41,769,600` 次样本-条件前向评估，历史 H200 测算约 26.5 H200·小时核心推理（排队、解码、I/O 另计）。这只是成本记录，不是已经测到的整批实际 walltime。
- **Stage 1 的 Utility Label 标准化是额外的一层可逆逐维标准化**，与冻结的 Stage 0 observation/action Normalizer 完全不同。最终用户选择全训练池 65,265 窗口，不再使用中途试验的 51,906/13,359 Student split；不声称存在独立 Validation Set。

---

## 4. Stage 1 **已实现**的网络结构（以 `stage1_core_architecture.md` + readiness 为准）

### 4.1 为什么采用并行 C0 + C1，而不是直接替换 Fusion

既有 Stage 0 的 `C0` 产生 `[B,256]` conditioning，旧 U-Net 已经习惯这个输入分布。为了在学习时序表示时不破坏已有策略，保留 C0 并行支路：

```text
                    7 路观测，每路动态传感器 2 帧
                                   │
                      冻结 Stage 0 五路 ResNet18
                                   │ 共用逐帧特征
                  ┌────────────────┴───────────────────┐
                  ↓                                    ↓
        旧 2-frame compression                 新逐帧 Frame Projection
         7 tokens + presence flags             + sensor/time embeddings
                  ↓                                    ↓
       原 C0 concat + MLP [B,256]          13 tokens [B,13,256]
                  │                                    ↓
                  │                        C1: 2×TransformerEncoder
                  │                         4 heads / FFN1024 / d256
                  │                                    ↓
                  │                           H [B,13,256]
                  │                                    ↓
                  │                    1 learnable query, 4-head AttentionPool
                  │                                    ↓
                  │                              z_t [B,256]
                  │                          ├─────────┬───────────┐
                  │                          ↓         ↓           ↓
                  │                     Adapter     Utility      Future
                  │                     256→256      Head        Predictor
                  │                          ↓
                  └─────────────────────── ADD ───────┐
                                              c_new = c0 + Adapter(z_t)
                                                        ↓
                                        Stage 0 Conditional U-Net
                                                        ↓
                                               Action noise prediction
```

- C1 的 **13 个输入 tokens，不含 CLS**：`front_t0,front_t1,side_t0,side_t1,wrist_t0,wrist_t1,left_t0,left_t1,right_t0,right_t1,state_t0,state_t1,language`；5 路 image ×2 =10，state×2=2，language×1=1。
- 与旧 7-token 路径的区别：旧路径先把**相邻两帧压成一个 token**；C1 新分支保留各帧独立特征。这仍然只使用 **2 帧原始观测**，不是 3 帧。旧 compression/C0 不变，共享 ResNet 提取，互不干扰。
- C1：hidden 256、2 Transformer Encoder layers、4 attention heads、FFN 1024、dropout 0。输出所有 `H [B,13,256]`，保留给未来 Stage 2 使用。
- **Attention Pooling 用一个 learnable query、4-head cross-attention**：Query 从 13 个 C1 output tokens 做加权聚合，得到 `z_t [B,256]`。没有使用 CLS，也没有多 CLS；一个 query 仍可通过多个 head 关注所有 token，不意味着只看一个模态。
- `Adapter = Linear(256,256)`，**weight/bias 全零初始化**，`c_new=c0+Adapter(z_t)`。因此刚初始化时 `c_new == c0`，同条件下 Stage 0 与 Stage 1 的去噪器和完整 100-step Action Prediction 可做到零误差。U-Net 接口仍是 **256D conditioning**，不是把 13 tokens 直接塞进 U-Net。
- 编码器/C0 初期冻结；新增 C1/heads 与 Adapter 分阶段训练，最后才低学习率解冻 U-Net。`W_adapt` 是额外 residual projection，**不是 Transformer 的标准内置层**。

### 4.2 两个 Auxiliary Heads（已实现）

**9D Utility Head**

```text
z_t [B,256] → Linear(256,128) → SiLU → Linear(128,9)
```

- 预测标准化空间中的 9D **signed regression**，不加 Softmax/Sigmoid、不裁掉负值。
- 逐维 pooled mean/std（标准差 `ddof=0`、floor `1e-6`），可逆映射回 raw utility。
- 默认 Huber(delta=1)，同时记录 standardized MSE，便于后续对比；维度组 Utility 不要求等于各单模态 Utility 直接相加。

**Action-conditioned Future Tactile Predictor**

```text
当前 z_t [B,256]
Demo Actions a_t...a_(t+3) [B,4,7] → 1-layer GRU(hidden=128)
每个时刻 concat(z_t, GRU hidden) [B,4,384]
共享 MLP 384→256→1024 + SiLU
→ predicted tactile latent [B,4,2,512]
```

- Frozen Stage 0 的左、右 ResNet18 编码未来真实触觉帧 `T_(t+1)...T_(t+4)`，取左右各 **512D raw latent** 作 target；使用 MSE，不训 target Encoder，不引入 target EMA。
- 因 Action Chunk 第 0 位为 `a_(t-1)`，Future GRU 必须选 **chunk index `[1:5]`**，即 `a_t...a_(t+3)`；对应未来 tactile `t+1...t+4`，且必须严格不跨 episode。
- **Demo Action 只能进入 Future Predictor 辅助分支**；未来触觉真实值只作监督；二者都不能流入 C1 或 DP Conditioning。否则构成训练目标泄漏。
- Future Loss 下降只能证明 latent 预测得到改善；是否真正使用 `z_t` 而非仅凭动作、以及是否建模物理接触动力学，尚缺 `P(z)`、`P(a)`、`P(z,a)`/Action Shuffle 等实验。

当前全 Stage 1 可训练新模块计 **3,818,633** 参数（含 Utility Head 34,057、Future Head 414,336 与 Adapter）；Warmup 时 Adapter 冻结，实际可训练参数 **3,752,841**。旧 U-Net 约 255,085,191 参数，最初冻结。参数量源于已上传代码报告，不代表其他变体架构。

### 4.3 训练目标和阶段切换原则

\[
\mathcal L_1=\lambda_A\mathcal L_{\mathrm{action}}+
\lambda_U\mathcal L_{\mathrm{utility}}+
\lambda_F\mathcal L_{\mathrm{future}}.
\]

- Phase 1：冻结 Encoder/C0/Adapter/U-Net，**只**训练 frame projection、C1、pooling、Utility Head、Future Predictor，**没有 Action 梯度**。
- Phase 2：解冻 Adapter，保持 U-Net 冻结；优化 Action+Utility+Future，让预训练 U-Net 通过 `c_new` 使用新表示。虽然 U-Net 参数冻结，**不得把其 forward 放在 `no_grad`**，否则 Action Loss 无法反传至 Adapter/C1。
- Phase 3：低 LR 解冻 U-Net 联合微调，仍冻结五路 Encoder 和 C0。切换时间不靠未验证的固定 300/300/1400 配方；阶段末保存并人工选择 Checkpoint，避免盲目续训。
- 因先前尝试发现梯度失衡，**不要把 Phase 1 的 `(0,0.1,0.1)` 或受控试验的 `(1,1.585e-4,1.585e-4)` 机械移植到 Phase 2/3**。新阶段重新测量各 Loss 对 C1 的梯度范数/方向，再决定辅助权重。

---

## 5. Stage 1 关键工程检查与早期受控训练

### 5.1 Readiness 检查（已完成，作业 381164）

最初只 Coding 了 C1+Adapter，**缺 Utility Head、Future Predictor 和 Label Join**；另发现 Stage 0 权重加载会遗漏动态创建的 Normalizer，Diffusion Scheduler 运行时状态在不同设备上导致假性的采样差异。均已修复。

- 严格恢复全套 E500 EMA / Normalizer，每个继承 key 必须匹配。
- 逐一核查 **65,265** 条标签与源码窗口身份；6D/7D Loss mask、Future offset 和轨迹边界正确。
- 确认全部四类 Head/Action 组合的梯度路由；冻结模块无参数梯度，零 Adapter 初始时 Action→C1 梯度为 0 是预期，而 Adapter 本身可收到梯度。
- **同观测、同 `k`/Noise 下，Conditioning、Denoiser、完整 100-step Action Prediction 的最大初始化差异均为 0**；没有通过放宽数值容差掩盖 Scheduler device 问题。
- 18 个固定窗 / 180-step Overfit：Action MSE **0.000833→0.000441**，Utility Huber **0.445145→0.004143**，Future MSE **0.525699→0.024174**。
- 另一次重新初始化后的独立 120-step Smoke（当时采用完整轨迹 split 的小探针）：Action **0.003506→0.003882（+10.73%）**，Utility **0.318→0.247**，Future **0.539→0.231**。**这些只是 small smoke 的诊断，不是最终 Stage 1 验证集成绩或闭环 SR**。

### 5.2 独立的 2,000-step Controlled Training（探索性试验，**不再作为正式训练的续跑起点**）

- 曾采用 300 steps U+F 预热、300 steps Adapter、1400 steps U-Net Joint，batch8、全量 65,265 窗口作为采样池、无 Val；**只访问 16,000 个不同窗口（24.52%）**。
- 诊断：Action MSE **0.002617→0.002930（+11.95%）**；Utility Huber **0.50549→0.36308（−28.17%）**；Future MSE **0.55658→0.17882（−67.87%）**。
- **Gradient Probe（重要）**：启用 Action 后，原 `λ_U=λ_F=0.1` 下辅助项在 C1 的加权梯度上界约为 Action 的 **630.84 倍**；因此在这条探索性训练轨迹里把两辅助系数按比例调至 **1.5851921885e-4**。这仅是当时特定状态下的梯度配平，不可视为新模型的普适最佳 Loss 权重。
- **Learning-rate Probe**：U-Net LR `1e-5` 时固定 Action Probe 曾增加约 **31.06%**；后续降低 U-Net LR 到 `1e-6`、Adapter 到 `3e-6`，最终回退减小至 **+11.95%**。这说明低 LR 候选更保守，**不证明 Action 最终能提升**。
- **Post-hoc U-Net 插值**：在不训练的情况下，`W=(1−α)W_teacher+αW_student`，仅插值 U-Net，试过 α=1/.75/.5/.25/.1/0；选择 α=.75 是在两组训练探针中 Action 增幅小于10%且保留尽可能多改动的规则，36/144 点分别 **+8.89%/+5.66%**。这不是最低 Loss，也不是仿真最优；`selected_inference.pt` **只有推理权重，不能精确续训**。
- TF32 probe：保持 FP32 tensor + TF32 kernels 相较严格 FP32 前后向更快（独立检验），不启用 BF16/FP16；端到端吞吐约 **2.41 steps/s**（workers6）且解码/I/O 是瓶颈。
- **关键决策**：这次 2k 是排查网络和超参的探索性实验。正式训练**回到 Stage 0 E500 EMA 从头初始化 Stage 1**，不从 `joint_step2000.pt` 或 α=.75 推理权重延续；改为较长的纯表征 Warmup，再人工决定是否解冻。

受控试验完整训练 Checkpoint：`outputs/stage1_controlled_2000_batch8_20261022_r4/checkpoints/joint_step2000.pt`，仅供复查，不是当前正式 Stage 1 主线。报告：`docs/stage1_controlled_training_results_20261022.md`。

---

## 6. **最新已执行**：Stage 1 Phase 1 Representation Warmup（10,000 steps）

### 6.1 执行设置与完成状态

- H200 **作业 `381592`，COMPLETED 0:0**，2026-10-22 17:41:23–18:12:14 CST，约 **30分51秒**。
- **全新 Stage 0 Epoch 500 EMA 初始化**，重置 AdamW/Scheduler，不继承 Controlled-2000 或插值候选任何参数更新。
- 全 65,265 窗口、不设 Validation；batch8，总采样80,000次，首个完整训练池覆盖发生在 step 8159 内（无放回排列，耗尽才重复）。
- 冻结所有旧 ResNet18、双帧压缩/C0、Adapter、U-Net 与 Normalizer；**Adapter 权重精确为零**，所有阶段始终保持原 Stage 0 Action Path。
- 只优化新 Frame Projection、Sensor/Time Embedding、C1、Pooling、Utility/Future Head，**3,752,841 trainable params**。
- Objective **`0.1 × Utility Huber(delta=1) + 0.1 × Future raw latent MSE`**；本阶段没有 Action Loss，且无需用之前 Action 梯度配平得出的 `1.585e-4`。
- LR=**1e-4 constant**、AdamW weight decay 1e-6、global grad clip 1、FP32 tensors+TF32 kernels、workers6。

### 6.2 关键 Loss / 预测探针

这里所有 Loss 均来自固定 **288 个训练窗口探针**（9任务×32窗），**不是验证集，也不是 Rollout**。

| Warmup step | Utility Huber | Standardized Utility MSE（仅诊断） | Future MSE |
|---:|---:|---:|---:|
| 0 | 0.383234 | 1.110066 | 0.543980 |
| 1000 | 0.236269 | 0.668670 | 0.076944 |
| 3000 | 0.218752 | 0.601761 | 0.040075 |
| 5000 | 0.206733 | 0.571288 | 0.033404 |
| **8000** | **0.195425** | **0.537942** | **0.028136** |
| 9000 | 0.201060 | 0.556547 | 0.029356 |
| 10000 | 0.200742 | 0.561513 | 0.028503 |

到 10k 相对于初始化：Utility Huber **−47.62%**、Utility MSE **−49.42%**、Future MSE **−94.76%**。8k 后边际收益减小，但**不能宣布严格收敛**。固定训练窗口上的改善不能推导 closed-loop Action SR。

Step 10000 部分 Utility Head 的**训练探针**预测 R²：V1 0.589、V2 0.653、V3 0.290、Left Tactile **0.438**、Right Tactile **0.476**、State 0.504、Language 0.256、Vision/group 0.680、Tactile/group 0.399。说明九维信号已初步可学习，但准确性不均衡。

Future 4 个 offset 的左右平均 MSE 约 **0.02090/0.03610**，远期 offset 更难。还没做 `P(z)` / `P(a)` / `P(z,a)`、Action Shuffle 或 Persistence 对照，**不能证明模型凭 `z_t` 学到了真实动力学**。

**局部异常**：Gear Assembly 左触觉 Utility 的探针 R²：8k **0.3386**、9k **0.2505**、10k **−0.1952**。只是每任务32个训练探针，不足以确认泛化问题，但提醒不要唯最新 Checkpoint。

### 6.3 重要的工程审计

- C1 梯度有限，整段均值 **0.05559**；冻结 540 个 Tensor **逐位未变**，新分支参数正常更新。
- 初始化及每 1000 steps，`c0/c_new`、Denoiser 相对于 Stage 0 的最大误差均为 **0**；直到 Warmup 结束 U-Net 仍与 E500 Teacher 行为完全一致。**Warmup 不是动作能力改进实验。**
- 完整 Checkpoint 含模型、Normalizer、Adam、Scheduler、全部 RNG、sample cursor、Utility 标准化统计、Teacher/label/source hash。初始化与每 1000 steps 保存，约 1.30GB/Checkpoint；GPU roundtrip 与精确恢复测试通过。
- 全套测试（报告口径）**320 passed / 3 skipped**，另有一个已知的独立 FiLM/CLIP 本地资源缺失失败；不影响本次主路径结论。

### 6.4 人工阶段切换决定（**已建议，但 Phase 2 尚未执行**）

- 8k：固定探针最优，但只访问64,000次、还未覆盖所有65,265窗口。
- **9k：已完整覆盖训练池，Utility 指标与 8k 接近，且 Gear Assembly Left R² 仍为正**；九任务中较差任务的预测质量略稳，是首选 Phase 2 起点。
- 10k：Future 进一步小幅改善，但 Utility 没有持续进步，个别任务回退；仅是最新，不应默认最优。

**当前主线选择：优先从 `warmup_step09000.pt` 进入 Phase 2**。这是人工作出的**训练工程选择，不是独立 val/闭环最优**；保留 8k、10k 供后续必要时复查，不能覆盖。

- Warmup 输出目录：`outputs/stage1_representation_warmup_ep500_batch8_10000_20261022/`
- 候选完整 Checkpoints：`checkpoints/warmup_step08000.pt`、`checkpoints/warmup_step09000.pt`、`checkpoints/warmup_step10000.pt`（相对于上述输出目录）。
- 曲线和诊断：同目录 `warmup_curves.png`、`warmup_prediction_quality.png`、`diagnostic_stepXXXXX.json`、`warmup_report.md`；详细依据 `docs/stage1_representation_warmup_results_20261022.md`。

---

## 7. 当前下一步：Stage 1 Phase 2 Adapter Adaptation（**还没运行**）

建议新 Session 直接从这里继续，而**不是**再次讨论是否生成标签、修改 K64、重训 Teacher、或重新进行 10k Warmup。

**建议执行方案（尚需 Codex 现场校验）**：

1. H200 从 Warmup **Step 9000 的完整 checkpoint** 加载模型/Normalizer/C1/Heads/Utility stats（从该点产生新的 Phase 2 分支）；旧 Encoder、历史压缩、C0、U-Net 保持冻结；**解冻 Zero-initialized Residual Adapter**。U-Net 可以参与可微前向，让 Action Loss 梯度经 `c_new` 返回 Adapter/C1，但 U-Net 参数不更新。
2. 目标改为 **Action + Utility + Future**。新模块 LR 暂参考 `1e-4`，Adapter LR 暂参考 `3e-6`；先在有非零 Action 梯度后核查**三个 Loss 对 C1 的加权梯度范数**，决定 `λ_U/λ_F`。**不能直接照搬 Warmup 的 0.1，也不要自动照搬 2k 试验的 1.585e-4。**
3. 先做约 **3,000 个 Phase 2 optimizer steps** 的受控实验，每约 1000 步保存完整可恢复 Checkpoint；关注固定训练点 Action Loss（相对 Stage 0）、Adapter/C0 输出 RMS、辅助 Loss、梯度稳定性、是否损害原动作路径，并检查优化器/阶段恢复逻辑。
4. 完成后**停下、提交报告**，由用户决定从哪个 Phase 2 Checkpoint 进入 Phase 3 U-Net 微调；**不要自动解冻 U-Net，不要自行启动 RTX Rollout 或长训**。
5. Phase 3 的 U-Net LR 可以把受控训练发现较保守的 `1e-6` 当作候选，但切换点、权重和正式训练预算**尚未批准**。不要将先前随口提出的 `80k/160k` 当成已经选定的实验方案或论文支持的最优 steps。

**无 Validation 的语义**：用户最终要求全 65,265 有效窗直接参与训练，主要是 Stage 0 过去的 Val Loss 从早期上升且与 Rollout 并不一致；这**不能证明 benchmark 一定有 bug**，也不等于模型没有过拟合风险。固定 Probe 是训练探针而不是独立泛化成绩。最终选模仍须在 RTX/4090 上运行统一协议的闭环 Rollout。

**现阶段明确没有的结果**：Stage 1 Phase 2/3 的训练曲线与最佳 Checkpoint；Stage 1 的 SR 改善；Utility Head 的 out-of-distribution 泛化；Future Latent 对交互动力学的因果贡献；Stage 2 的任何训练/消融结果。

---

## 8. 核心科学问题、文献依据及下一阶段注意事项

- **时序建模**：旧 C0 将同路双帧早期压成 1 token，新 C1 保留 13 个 per-frame sensor/state/language tokens。使用的 **2 层 Transformer + 单 learnable-query Attention Pooling** 是当前工程候选，不应称为被 MS-Bot 证明的最优结构。
- **为何保留 Future Loss**：仅用 Action Loss 可能偏向容易拟合的状态特征；Future Tactile Latent 作为辅助监督，旨在使 `z_t` 保留与接触变化相关的动态信息。但 `P(a_demo)` 本身也可能拟合 target，需要 `P(z)`、`P(a)`、`P(z,a)` 或动作打乱验证。与 AdapTac-Dex 等先前研究存在相关先例。
- **MS-Bot 对照关系**：MS-Bot 的动态融合涉及 Action History、状态/阶段 Token 与 Cross-Attention，**不是**我们目前的单 CLS/C1 实现；后续若复现应按原论文协议。其 Stage Supervision 与我们**连续 Signed Utility 监督**的目标不同。
- **Future supervision 位置**：Representation-Aligned Tactile Grounding 等工作启发未来对 C1 `z_t` 与动作 Backbone 中间表征做 Probe，但 Stage 1 已实现的第一个版本把 Future Loss 加在 `z_t` 分支上，不应冒称已完成层间消融。
- **Stage 2 设计仍待细化**：计划把 `H[B,13,256]`、`z_t[B,256]`、预测 `u_hat[B,9]` 用于 Query 调制（residual/attention bias 等）和 KV 端 FiLM/HyperNet，再用 C2 融合。不能把 Stage 2 动态融合的收益预先归因于 Utility；需要同容量 state-only 对照和相同输入/训练预算。
- **Teacher 偏差**：Stage 0 后期见过所有 demonstration，离线 label 是对这个特定 Teacher 的 in-sample sensitivity；尤其 Object Search 策略失败，不能把其高 utility/时序曲线解释成已掌握任务。Group mask Utility 与 individual Utility 非线性相关，不能简单相加当成公理。

---

## 9. 交接给新 Session 的第一条消息（可直接复制）

> 请先阅读附件 `AdaViP_Session_Handoff_2026-10-22.md`，以其中第 4–7 节和我提供的 H200 `docs/` 为准。当前 ManiFeel 的 Stage 0 E500 EMA Teacher 和全 65,265×9 K64 Labels 均已冻结/就绪，Stage 1 C1（13 Temporal Tokens、2 层 Transformer、单 Query Attention Pooling）、C0 残差 Adapter、9D Utility Head、Future Tactile Predictor 均已实现。独立正式训练 Phase 1 已在 H200 完成 10k steps，推荐 **Warmup Step 9000** 为 Phase 2 起点，但 Phase 2 尚未执行。下一步先协助审查与设计 **Adapter Adaptation**：冻 U-Net、解冻 Adapter、引入 Action Loss 并重新测辅助梯度配比；约 3k steps 受控训练后停下报告。不要将 2000-step 调参试验视为正式训练前缀，不要自动长训或开始 Stage 2。所有 H200/RTX 指令请指明各自用途，简洁回答。

---

## 10. UniVTAC 第二 Benchmark：Stage 0 已结束，结果与 Teacher 候选

> **这节仅记录 UniVTAC Stage 0 的现状与已提供的评估结果；尚未建立 UniVTAC 的正式 Utility Label Protocol，也未启动它的 Stage 1。** 来源为用户上传的八任务×六 Epoch×每任务 10 次 Rollout 截图及本轮会话；暂无进一步 RTX 扩大评估结果，不把先前“建议去跑”写成已完成。

- 仓库：<https://github.com/ggkiller-air/UniVTAC>。
- Stage 0：八任务统一 Multi-task DP，按已有仓库文档使用 Head + Wrist RGB、左/右触觉、Proprio 和固定 CLIP 任务嵌入。与 ManiFeel 一样计划采用独立传感器编码、full/removal、统一 Teacher；UniVTAC 只有**两路相机**，因此后续删除模态 Utility 预计是 **8D（6 个输入分量 + 2 个 group）**，具体必须以当前代码/Mask 顺序核验，不能照搬 ManiFeel 9D。
- 已训练并测试 E100、E200、E300、E400、E500、E600 六个 Checkpoint；每 Epoch 八任务各 **10 次 Rollout**，共 80 次。下表是成功次数/10：

| UniVTAC Task | E100 | E200 | E300 | E400 | E500 | E600 |
|---|---:|---:|---:|---:|---:|---:|
| Insert HDMI | 0 | 1 | 2 | 3 | 1 | 1 |
| Put Bottle in Shelf | 3 | 3 | 4 | 5 | 5 | 5 |
| Lift Bottle | 2 | 4 | 8 | 9 | 6 | 7 |
| Pull-out Key | 4 | 8 | 8 | 7 | 6 | 6 |
| Grasp Classify | 10 | 10 | 10 | 10 | 10 | 10 |
| Lift Can | 1 | 1 | 1 | 2 | 4 | 6 |
| Insert Hole | 2 | 6 | 4 | 6 | 4 | 4 |
| Insert Tube | 4 | 8 | 9 | 9 | 7 | 6 |
| **成功总数 / 80** | **26** | **41** | **46** | **51** | **43** | **45** |
| **Macro SR** | **32.50%** | **51.25%** | **57.50%** | **63.75%** | **53.75%** | **56.25%** |

**当前 Teacher 候选：Epoch 400（51/80 = 63.75%）**，但每任务仅 10 次 Rollout，无法认定显著优于 E300/E600；现阶段**没有正式冻结 UniVTAC Teacher 的决定**。用户已讨论让 **RTX/4090 仿真服务器** 对 E300/E400/E600 采用相同环境初态扩大每任务约 30 次 Rollout，再选统一 Teacher。此建议截至本文制作时**不是已执行结果**。之后 UniVTAC 可以依次走 Teacher 选择 → 离线 8D Utility 采样精度诊断 → Pilot → 全量 Label → 共用 Stage 1/2 主结构的路线；但该流程**尚未开始**。
