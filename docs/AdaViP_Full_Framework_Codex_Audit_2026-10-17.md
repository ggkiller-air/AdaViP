# AdaViP 全框架设计规范与 Codex 只读审查任务书

**Version:** 2026-10-17 / Full-framework audit v1  
**Research target:** *AdaViP — Utility-Supervised Adaptive Visuotactile Perception for Robotic Manipulation*  
**Repository:** https://github.com/ggkiller-air/AdaViP  
**Status:** 研究架构候选方案；**未完成新方案训练与验证**。本文件不是已通过的工程实现规范。  
**Scope:** Stage 0 → Utility Label Generation → Stage 1 → Stage 2 → Evaluation/Ablation。  
**Core instruction:** **本轮只审查，不修改代码、不创建训练任务、不启动 GPU 长训、不擅自提交/推送、不新建分支。**

> **发给 Codex 的开场指令**  
> 你现在是 AdaViP 的 research-oriented code auditor，不是立即编码的实现代理。请认真阅读本文件及仓库中已有 ManiFeel Multi-task DP、传感器加载、HyperNet、实验日志和 checkpoint 代码。以「文件路径 + 行号/配置键 + 实际 tensor shape + 可复现实证」回答问题。必须把事实、设计建议和尚待验证的假设严格区分；禁止凭经验虚构数据字段、训练超参、模型效果。先提交完整审查报告和可实施的分阶段修改方案，等我确认后才进入编码。尤其**不要简单复用旧 Multi-task DP**：旧代码没有使用全部目标传感器，只能作为数据、环境、动作接口及训练入口的参考，新的 full-sensor reference policy 预计需要较大改造。

---

## A. 阅读范围、术语和当前决策

### A1. 材料优先级

1. 本文件的**最新架构决定**优先于旧 session handoff 中的过时设置。
2. 用户两页手绘图 `BO1.pdf` 是架构构思依据，不代表某个层已具备可直接实现的 tensor 接口。
3. 旧交接记录 `AdaViP_Research_Discussion_Handoff_2026-10-17(1).md` 提供研究动机、参考文献和先前权衡，其中 **5D/7D utility、proprio 始终保留、单任务 Stage 0、Stage 1 不含 action loss** 等旧决策已经被下面最新设置取代。
4. 真正的路径、keys、mask、模型版本、checkpoint 兼容性等必须以当前仓库和实际 dataset 为准；官方论文只证明论文声称的 benchmark 配置，不能证明本地数据已正确加载。

### A2. 已确定的研究设计（原则层面）

- **ManiFeel simulation multi-task**：统一训练一套 task/language-conditioned DP；优先覆盖实际可用的全部目标任务，而不是逐任务分别训练 Teacher。
- **7 个基本输入 token**：3 cameras (`V1,V2,V3`) + 2 tactile fingers (`T1,T2`) + proprioception (`P`) + task instruction/language (`L`)。每路 **sensor-level 一个 token** 是第一版的目标，非 camera patch-level utility。
- **9D signed utility**：前七维为单路 input removal 的 loss difference，再加 `all vision removed`、`all tactile removed` 两个 group utility。
- **Stage 0**：普通 full-sensor multi-task **Reference Diffusion Policy**；只有 action/denoising loss；训练过程带 single-input 和 group-input dropout；不含 HyperNet、Utility Head、Future Prediction、第二级 Transformer。
- **Utility label generation**：在 Stage 0 teacher 训练、选定并冻结 checkpoint 后，用 paired denoising loss 比较 Full 与各 Mask Condition；可以离线缓存，也可以在学生训练前按需批量生成，但不得依赖仍在变化的 Teacher。
- **Stage 1**：多模态 Encoder + 第一级 State Compression Transformer `C1` → `z_t`；同时训练 action DP、9D Utility Head 和 **action-conditioned Future Tactile Latent Predictor**，联合监督但采用渐进式启用与独立消融。
- **Stage 2**：原则上冻结 Stage 1 的 Modality Encoders + `C1` + Utility Head；加入按模态共享的 HyperNet-FiLM、Utility-aware Query、第二级 Adaptive Fusion Transformer `C2`；输出 `z'_t` 给 DP。Stage 2 可微调 DP action backbone；必须有初始恒等路径以免毁掉 Stage 1 已学策略。
- **方法假设尚未验证**：9D utility 能否提升 action performance、未来触觉辅助目标能否提供有用的 interaction representation、HyperNet/Query 两处利用 utility 是否互补，均需单独实验。

### A3. 两个不能偷换的概念

- `sensor-level utility` ≠ `pixel/patch-level utility`。一个相机输出 1 个 policy token，删除该 token 得到的是相机级重要性；即使 FiLM 调节向量各 channel，也不能宣称已有 patch-level supervision。
- `leave-one-out loss increment` ≠ Shapley value ≠ 真实因果效应 ≠ robot rollout success contribution。该分数仅反映**固定 Teacher 与其训练 loss**下，特定 mask 的条件损失增量。

---

## B. End-to-end 架构：必须审查的信息流

### B1. 符号与预期 shape（`B, d, H` 均为占位；须按代码确认）

| 符号 | 作用 | 目标形状（示意） |
|---|---|---|
| `V1,V2,V3` | 3 路 RGB camera | `[B, obs_horizon, 3, H, W]` 每路 |
| `T1,T2` | 左右 tactile observation；**TacRGB/TacFF 选型需确认** | `[B, obs_horizon, ...]` 每路 |
| `P` | proprioception | `[B, obs_horizon, p_dim]` |
| `L` | task instruction embedding 或 task-conditioned language token | `[B, d]` |
| `F_t = [f_{v1},f_{v2},f_{v3},f_{t1},f_{t2},f_p,f_l]` | 对齐到同一 `d` 的 per-input tokens | `[B, 7, d]` |
| `z_t = C1(F_t)` | compact interaction-state token | `[B, d]` |
| `u_hat_t = U(z_t)` | 9D signed teacher-derived utility estimate | `[B, 9]` |
| `F'_t` | FiLM 后 7 路 sensor/language tokens | `[B, 7, d]` |
| `q_t` | Utility-conditioned Query | `[B, d]` |
| `z'_t` | 第二级 residual adaptive fusion output | `[B, d]` |
| `a_{t:t+H-1}` | DP action chunk | `[B, pred_horizon, action_dim]` |

**必须审查**：camera/tactile 观测是否同频同时间戳；`obs_horizon` 是否跨多个物理时刻；同一传感器多时间帧如何编码为一个 token。`token-level` 指上述 7 路 input token，不是 Transformer 内的任意 patch。

### B2. Stage 0：Reference Policy

```text
Full observations: 3V + 2T + P + L
          ↓
  7 input-specific encoders / embeddings
          ↓
  sensor-aware mask + explicit mask indicator
          ↓
  simple fusion (concat / light MLP / minimal attention; evaluate cost)
          ↓
  diffusion action policy
          ↓
  L_action (same target/noise scheduler across configurations)
```

此处**不加入** `C1`、`C2`、Utility Head、HyperNet 或 Future Tactile Predictor。旧 Multi-task DP 的 full-sensor input path 并不完整，必须按实际结构决定重构范围，而非只改一个配置开关。

### B3. Stage 1：学习可供 Stage 2 冻结的 `z_t`

```text
Full observations ──> modality encoders ──> 7 aligned tokens
                                                ↓
                                     C1: State Compression Transformer
                                                ↓
                                               z_t
                                 ┌──────────────┼──────────────┐
                                 ↓              ↓              ↓
                             Utility Head   Action DP    Future Tactile Head
                               9D u_hat      L_action     P(z_t, demo actions, Δ)
                                 ↓                             ↓
                         L_utility vs labels            L_future vs future latent
```

**设计预期**：Utility/Future 的梯度能够实际进入 `C1` 和指定可训练的 modality encoders；Future Predictor 只在训练中出现，不修改推理路径。不得因将 Teacher 的完整 checkpoint 强行加载到新 `C1` 架构而造成 shape mismatch：可以只移植兼容 encoder/DP 参数，详见审查任务。

### B4. Stage 2：Utility-guided Adaptive Fusion

```text
Full observations ──> Stage-1 modality encoders [frozen] ──> F_t
                                                 │
                                                 ├──> C1 [frozen] ──> z_t
                                                 │                    ↓
                                                 │               U [frozen] ──> u_hat(9D)
                                                 │                    │
                                                 └──────> per-modality HyperNet-FiLM
                                                                   ↓
                                                            F'_t (7 tokens)
                                q = Utility-conditioned(z_t, u_hat)
                                                  ↓
                                C2: Cross-Attention(Q=q, K/V=F'_t)
                                                  ↓
                                z'_t = z_t + residual_adaptation
                                                  ↓
                                    initialized/finetuned Action DP
```

此处 `C2` 是**另一个 Transformer block / cross-attention mechanism**，不是把 `C1` 输出直接替换。**`C1` 与 `C2` 参数不共享**；若拟用 `C1` 初始化 `C2`，须列出 self-attention → cross-attention 的可复制权重和不可复制权重；不强制复制。

---

## C. 9D Counterfactual Utility：精确定义和 Teacher 要求

### C1. 9 个干预条件

给定输入索引 `V1,V2,V3,T1,T2,P,L`，令被删除集合 `S` 的 loss 为 `L_ref(F^{-S}; ε, k, a*)`，其中 `ε` 是 diffusion noise，`k` 是 diffusion timestep，`a*` 是同一 demonstration action chunk。

\[
 u_S^*(t) = \mathcal L_{ref}(F_t^{-S};\epsilon,k,a^*) - \mathcal L_{ref}(F_t;\epsilon,k,a^*).
\]

| label index | mask condition | 含义 |
|---|---|---|
| 0 | `{V1}` | Camera 1 contribution |
| 1 | `{V2}` | Camera 2 contribution |
| 2 | `{V3}` | Camera 3 contribution |
| 3 | `{T1}` | Left tactile contribution |
| 4 | `{T2}` | Right tactile contribution |
| 5 | `{P}` | Proprio contribution |
| 6 | `{L}` | Task instruction contribution |
| 7 | `{V1,V2,V3}` | Entire vision-group contribution |
| 8 | `{T1,T2}` | Entire tactile-group contribution |

需 **1 full + 9 masked = 10 teacher forward-equivalents / paired noise sample**；可以批量 vectorize，但算法意义上不是 9 次。**不能预期 `u_V = u_V1+u_V2+u_V3` 或 `u_T=u_T1+u_T2`。**

### C2. 训练 dropout 必须和标注干预匹配

- Teacher mask 分布需**显式覆盖上述 9 类**，再加 full condition；**不需要覆盖全部 `2^7` 组合**。
- 高比例 full condition、低但非零频率每类 group mask 是**候选原则**，不是已确定概率。Codex 必须提出具体分布候选和预估每条件样本覆盖率，说明是否会明显降低 Full-input policy performance。
- Mask 语义与训练/标注一致；建议在 encoder 后、fusion 前处理，**同时给出 mask identity/indicator**。若使用 zero replacement、learnable missing embedding、attention mask 等，请给出优缺点和 padding 兼容性。
- `P` 与 `L` 也必须在 Teacher 训练中可被移除，不能沿用旧 handoff “P always present” 的设置；但这两项遮蔽可能带来不可恢复的信息缺失，应作为重点风险监测。
- 遮蔽 `L` 的作用可能主要是 task identification；不要将其误解释为接触阶段的语言 utility。遮蔽 `P` 可能显著损害控制，应评估其标签分布和任务差异。

### C3. Paired diffusion evaluation 不可违背的不变量

对 `full` 与全部 mask，固定并复用以下要素：

1. 同一 demo sample / task ID / observation window / action window / target。
2. 同一 diffusion timestep `k`、同一 noise `ε`、同一 noisy action `x_k`、同一 denoising target。
3. 同一 action normalization、valid action-dimension mask、loss reduction 规则。
4. 同一评估模式（`eval()`）、无随机 sensor augmentation；若有非确定因素必须做成对控制。
5. **保留 raw signed differences**（负值可能存在）；为训练另存 train-split-only 的标准化统计，不能在生成 raw 标签时截断所有负数。
6. 可评估 `K` 个 independent `(ε,k)` 重复求均值以降低 variance；`K` 待基于 pilot label variance / compute cost 选定，不预设唯一正确数值。

尤其不要将单次 diffusion denoising loss difference 写成**真实动作成功率提升**。必须报告 noisy label 稳定性与 per-task 分布。

### C4. 数据标注内容与泄漏

Label cache 至少保存：`task_id`、`trajectory_id`、`timestep/index`、teacher commit/checkpoint、dataset version、9D raw signed labels、每个 mask 的 absolute loss、full loss、loss reduction、随机种子/噪声处理、normalization stats/split、各模态有效性标记。时间窗口不得跨越 episode 边界。

---

## D. Stage 0 只读审查：首先验证“多任务+全部传感器”是否真正可实现

### D1. 数据与输入清点（最高优先级）

官方 ManiFeel Appendix Table V 报告仿真任务记录 `3 Cam / 2 Tac / 7D Proprio`，但**不能假定每个本地样本包含可直接拼接的 7 个 token**。请读取仓库 dataloader 和本地实际样本（若有访问权限）：

- 逐任务列出真实 dataset keys、shape、dtype、数据量、episode 数、采样频率、camera view identity、tactile type (`TacRGB` vs `TacFF`)、proprio 字段、action 字段。
- 明确相机顺序、left/right tactile 的身份，是否存在可见性或 task-specific 缺失；摄像头是否已在旧代码里按 task 选掉一部分。
- 检查 observation timestamp、action timestamp、采样频率、`obs_horizon` / `action_horizon` / `pred_horizon`、未来触觉样本的索引边界。
- 官方 Table V 含 **6D 和 7D action tasks**，明确每个任务的 action 语义、控制模式、旋转参数、gripper 维度，给出**不破坏物理语义**的统一方案以及 loss mask / action normalization 策略。**仅 zero-pad 到 7D 且没有 valid-dim mask 不算审查通过。**
- 官方论文 Appendix 列出 *Ball Sorting (Normal)* 与 *(Dim)* 分别配置。请核查本地究竟是 9 个任务、10 个配置，还是其它集合；**不要凭项目名断言是固定九个**。
- 检查 train/validation/test 是否按 trajectory 分隔，避免同一轨迹窗口泄漏；multi-task dataloader 如何平衡任务及不同 action statistics。

### D2. Language / Task-conditioning

官方数据摘要没有将 language 作为原生传感器字段列出。若本地无 language：

- 给每个任务/配置建立固定的自然语言指令或 task ID 对应的语言表达；确认相同任务的多种指令是否存在，未存在则不能宣称 language generalization。
- 比较 `learned task-ID embedding`、`frozen text encoder + fixed prompt` 在算力、参数、未来迁移方面的代价，提出**最简单可靠的首选**，不得擅自加载庞大 VLM。
- 语言单任务内恒定，不代表语言 utility 必为零；在多任务中遮蔽语言可能降低 task discrimination，但这一信号未必代表接触状态。
- 如果 `task_id` 被用于 loss masking、数据归一化、action decoding 等，请检查它是否通过旁路让语言遮蔽变得“伪遮蔽”；列出哪些元信息运行时确实可用、哪些会产生 task-ID leakage。

### D3. Teacher 设计及训练稳定性

- 查清历史 Multi-task DP：哪些组件能复用（训练器/采样器/scheduler/eval），哪些因只有部分传感器而必须重写（dataset / encoder / fusion / masks）。列出**具体文件级别改造图**，不得建议原样继续用旧 checkpoint 当 full-sensor Teacher。
- 建议 Simple Fusion 范围：同模态 Encoder 是否共享、对齐 token 维度、最小 cross-sensor fusion 的选择；保持足够表达能力但避免引入 Stage 1/2 的研究模块。
- 检查 DP action loss 为 epsilon-prediction、sample-prediction 或 v-prediction、noise schedule、action-horizon 和动作有效维度；训练/labeling 应完全一致。
- 设计 dropout policy，比较 `no-dropout full-sensor DP` 与 `sensor-dropout full-sensor DP` 的 per-task success、full-input validation loss、masked condition loss；不能只看训练 loss。
- 建议 balanced multi-task sampling、grad norm、学习率 warmup、EMA、混合精度，**分别说明来源：旧配置、论文、还是提案**；不要伪称为官方最优。

### D4. Stage 0 gate / 中止条件

在 Stage 0 正式长训前需用少量 batch 确认：7 路信息确实进 network；Mask 作用于预期位置；梯度传递正常；6D/7D loss mask 正确；各数据模态在 episode 内对齐。短训后再确认 full-input baseline 不崩、各任务没有完全失败；dropout teacher 的 masked loss 不随机爆炸；若某模态 utility 全为零或始终巨大，先分析 Teacher 与遮蔽机制，不进入 Stage 1。不要为达成主方法效果而人为设定“必须高 utility”。

---

## E. Stage 1 审查：Action + Utility + Future Tactile 联训

### E1. 模型目标

\[
F_t=E(o_{t-k:t}, l),\quad z_t=C_1(F_t),\quad \hat{\mathbf u}_t=U(z_t)\in\mathbb R^9.
\]

\[
\mathcal L_{Stage1}=\mathcal L_{action}+\lambda_u\mathcal L_{utility}+\lambda_f\mathcal L_{future}.
\]

- `C1` 必须明确：输入 `[B,7,d]`，是 `CLS/state query` 聚合、可学习 state token 还是 pooling；具体深度/head/dim 需代码与小规模实验决定。避免无理由堆大 Transformer。
- **只在 Stage 1 完成且动作性能和辅助监督都验证后，Stage 2 才冻结 `E/C1/U`。**
- Stage 1 action policy 直接消费 `z_t`。如 Stage 0 simple fusion 输出维度/分布不同，须明确哪些参数可移植、哪些不能移植；不能假定 Stage 0 的 DP action backbone 与 Stage 1 完全兼容。
- Utility Head 预测 9 个**实值 signed score**；不施加 Softmax 或非负性约束。先比较 MSE/Huber 及统一 vs per-dimension/per-task normalization；注意任务差异、label noise、imbalance 和巨大的 `u_P/u_L` 导致其它维度梯度被淹没。

### E2. Future Tactile Latent Prediction（预定最小候选）

**第一版候选**：使用 `z_t` 和来自当前轨迹的**真实 demonstration action subsequence**，预测 `t+Δ` 两路未来触觉的**latent targets**：

\[
\hat e^{(i)}_{t+\Delta}=P_{\phi}(z_t, a^{demo}_{t:t+\Delta-1}, \Delta, e_i),\quad i\in\{left,right\}
\]

\[
e^{(i)}_{t+\Delta}=E_T^{target}(T^{(i)}_{t+\Delta}),\qquad
\mathcal L_{future}=\operatorname{mean}_{i,\Delta}\|\hat e^{(i)}_{t+\Delta}-\operatorname{sg}(e^{(i)}_{t+\Delta})\|^2.
\]

约束：

- `demo actions` **只能进入辅助预测 head**，绝不能进入 `z_t` encoder、DP action prediction 的观测路径或部署时的数据流。
- `Δ` 的选择必须受 action chunk 与实际 episode/time index 限制；不能从训练窗口错位地取未来触觉，不得跨 trajectory 边界。
- `E_T^target` 首先审查是否能够用固定（frozen）并且已经训练过的 tactile encoder 生成稳定 latent。EMA 是可评估备选，不能因为有 `stop_gradient` 就保证不 collapse。
- 明确 2 个 tactile fingers 共享/不共享 predictor、`latent d`、loss scale、训练时 predictor 参数是否保留于 policy checkpoint，推理时是否完全移除。
- **必须做 `action-only P(a)` 对照**，检验 future target 是否可以几乎完全由 demo action 预测，使梯度对 `z_t` 无益；至少评估 `P(z_t)` 与 `P(z_t,a)` 的 latent prediction error 和下游 action performance。
- **该结构是 DP 上的改编提案，不是论文已验证的标准 DP 配方。**参考的 *Representation-Aligned Tactile Grounding* 在 VLA 的 action-expert intermediate features 上预测 future tactile latent；它的最优中间层结论不能直接迁移到 DP。若有余力，在 Stage 1 后用 frozen probes 比较 `z_t` vs DP intermediate features，再决定是否移动预测头。
- 不把预测 future vision 作为第一版必要目标，以免给实验增加额外变量。

### E3. Gradients、目标冲突和训练顺序

必须确认 `L_action`、`L_utility`、`L_future` 的 backward 对每个 `E/C1/U/P/DP` 模块的梯度路径。专门检查：

- Future Head 对 `C1` 有梯度，但无法向 action supervision 偷运未来 observation / action。
- Utility Target 无教师可训练梯度（Teacher fully frozen），未混入 Student 在线输出。
- 维度不平衡或频繁大 utility 使 `L_utility` 压制 action loss；未来 latent 范数过大使 `L_future` 主导训练。
- 对 shared representation 记录每类 loss 的梯度范数、cosine similarity（可选诊断），监测相互抵消和 optimization instability。

**执行顺序建议（不是改变最终三损失目标）**：

1. Stage 1 `Action only` 作为 architecture-matched baseline；
2. `Action + Utility`，验证 9D label predictability；
3. `Action + Future`；
4. `Action + Utility + Future`；
5. 再依据表现决定是否在 full run 使用不同 loss warmup/curriculum。

**不得**直接启动一个复杂 three-loss 大训练且只保留最终 success number。

---

## F. Stage 2 审查：Frozen State + Utility-aware Adaptive Perception

### F1. 冻结策略

第一轮基线：`E`、`C1`、`U` 冻结；`H_V,H_T,H_P,H_L`、FiLM 与 `C2` 训练；DP 从 Stage 1 兼容 checkpoint 初始化并小学习率微调，是否需要冻结 DP warmup 由审查建议决定。

- 不能只冻 `C1` 而放任前面 `E` 改变分布，否则 `C1/U` calibration 仍会漂移。
- 用 Stage 1 representation checkpoint，而不是默认 Stage 0 Teacher checkpoint。
- 若后期解冻 `E/C1`，需考虑恢复 Utility Loss/校准项并与冻结版本对照。

### F2. HyperNet-FiLM：同模态共享网络，各 token 输出不同参数

7 路 token 对应四个模态：`Vision`（3 sensor tokens）、`Tactile`（2 sensor tokens）、`Proprio`（1 token）、`Language`（1 token）。优先设计四个**轻量** modality-specific HyperNet，且同一模态内跨 sensor **共享网络权重**；如果 Proprio/Language 的独立 HyperNet 参数成本太高，可提出轻量替代，但不能擅自改动研究含义。

对 `i` 属于 Vision：

\[
(\gamma_i,\beta_i)=H_V(z_t,\hat u_i,\hat u_V, e_i),
\quad f'_i = f_i + \underbrace{[\gamma_i\odot f_i+\beta_i]}_{\text{residual FiLM, identity init}}.
\]

对 Tactile 用 `H_T(z_t, u_hat_i, u_hat_T, e_i)`；对 Proprio/Language 使用对应的 `u_P/u_L`。若使用 `f'_i=(1+γ_i)⊙f_i+β_i`，需说明与上式等价及初始化方式。

- 符号中的 `e_i` 是固定传感器身份 embedding（不同 camera/tactile 左右），避免共享 HyperNet 生成相同调制。
- 注意 group utility 与 local utility 的冗余及量纲差异。建议前置标准化 + 小 projection，训练时梯度是否需要经过 `U`（第一轮 frozen，无需）。
- `gamma/beta` 必须初始化为近零残差；不能让随机 HyperNet 大幅扭曲 Stage 1 encoder features。
- 不能把 FiLM `γ,β` 自然解释为注意力概率或可观测的因果贡献。

### F3. Utility-aware Query：借鉴 MS-Bot 的思路，而非照搬 Stage Labels

候选：

\[
q_t=\operatorname{LN}\big(z_t+\alpha G(\hat{\mathbf u}_t)\big),
\quad Q=W_qq_t,\quad K=W_kF'_t,\quad V=W_vF'_t.
\]

- `G` 将 9D utility 投影到 `d` 维，`α` 为可学习/受控的 residual gain；**具体是否需要 LayerNorm、α 的初始化和 parameterization 应由代码审查决定**。如果需保证严格初始 `z'_t=z_t`，应在融合输出 residual 处作恒等初始化，不能错误地假定 Query 经过 LN 就等于 `z_t`。
- 按 MS-Bot 的启发利用 state 与辅助 condition 形成 Query，但我们的 utility **不是人工 stage label**。Code auditor 应对照 MS-Bot 论文实现解释这一 adaptation 的相同点和差异。
- `Utility → HyperNet` 与 `Utility → Query` 是两条使用路径，必须单独消融；不能仅因架构看起来多处使用 utility 就声称更合理。

### F4. C2 与零退化初始化：必须优先确保不崩

\[
\delta_t = C_2(Q=q_t,K/V=F'_t),\qquad
\boxed{z'_t=z_t+g\cdot\delta_t},
\]

其中 `g·δ_t` 在 Stage 2 第 0 step 应为**精确零或数值上很小**，即初始化输出应接近 `z'_t=z_t`，以兼容 Stage 1 DP。注意若 `g=0` 且部分路径被设计成永远无梯度会卡死训练；Codex 必须明确可学习 gate 与 output projection 零初始化的梯度是否可达，以及第一步哪些参数有非零梯度。

**强制测试**：对同一个 batch，启用 Stage 2 新模块但未训练时，Stage 1 与 Stage 2 的 `z_t` / `z'_t`、DP denoising loss、fixed noise prediction 应在定义的容差内一致（并记录容差和随机性控制）。这是一条阻断性验收条件。

- 第二级 Cross-Attention 的 head/dim、PreNorm/PostNorm、position/sensor embeddings、是否包含 proprio/language 的 K/V 都需明确。
- `C2` 可从头训练并保留 identity residual；也可部分从 `C1` 初始化，但不要错误地全量 `load_state_dict(strict=False)` 后宣称成功复用。
- 如果 `z'_t` 和 `z_t` 的尺度不一致，将导致 DP conditioning shift；必须监控 token norms、attention distribution、FiLM parameter distribution。

---

## G. 必须执行的实验与否证条件（请先审查可实现性，不要现在启动）

### G1. Stage 0 baselines

- Full-sensor Multi-task DP **without dropout**。
- 同架构 Full-sensor Multi-task DP **with planned dropout**。
- 评估 per-task rollout success、mean ± variability over seeds（预算允许时）、full-condition validation denoising loss、9 masked-condition loss。

### G2. Stage 1 最小对照（同 Student 架构、同数据与训练预算）

| 组别 | Action | Utility | Future tactile | 核心问题 |
|---|---|---|---|---|
| S1-A | ✓ | — | — | `C1` 自身是否稳定 |
| S1-B | ✓ | ✓ | — | 9D supervision 有何收益 |
| S1-C | ✓ | — | ✓ | Future objective 是否有收益 |
| S1-D | ✓ | ✓ | ✓ | 联合监督是否互补 |

记录不仅是 action success，还包括 utility prediction MAE/MSE、每任务每维 utility correlation、future latent prediction error，以及是否出现梯度冲突/表征崩塌。

### G3. Stage 2 必需消融

| 组别 | Utility supervision in pretraining | HyperNet-FiLM | Utility in Query | 用途 |
|---|---|---|---|---|
| S2-0 | matched | — | — | 无适配、但有相同第二层容量的 baseline |
| S2-1 | matched | State-only | — | HyperNet 不用 utility 的对照 |
| S2-2 | matched | — | ✓ | 仅 Utility-conditioned Query |
| S2-3 | matched | ✓ | — | 仅 Utility-conditioned FiLM |
| S2-4 | matched | ✓ | ✓ | Full AdaViP |

额外关键比较：`Utility Head有结构但不使用 teacher supervision` vs `带显式 teacher supervision`，确认改善不是多几个参数/conditioning path 造成的。比较 C2 仅作为 adapter 的提升而非硬把新 Transformer 的 capacity 当成 utility 贡献。

### G4. 可视化 / 解释性

- 同一轨迹的 9D **raw signed utility** 时间曲线 + 每任务分布 + 与接触变化的对齐（如果真实接触事件标签存在）。
- Utility prediction vs teacher targets（scatter / correlation / calibration），尤其 V/T group 与单 sensor 的联系。
- HyperNet-FiLM `γ/β` norm、attention scores、对 utility 变化的响应；**Attention 不等于解释性证明**。
- Teacher/student/final policy 实际逐路 sensor removal sensitivity 与 predicted utility 是否一致；若失配，应明确报告 teacher-specific bias。

### G5. 明确判定失败的情况

- Full-sensor reference DP 比合理 baseline 显著劣化，或多任务无法收敛。
- Dropout 导致模型实际上只靠某几路传感器，很多 utility 维度接近常数/纯噪声。
- Stage 1 utility regression 很好，但 action performance 无增益甚至下降；不能继续声称「action-relevant utility 有效」。
- Future Head 完全靠 action 预测未来 tactile，`z_t` 表征没有受益。
- Stage 2 一接入第二层就破坏初始 DP，而不是从 Stage 1 等价初始化。
- 加了 utility conditioning 与同容量 state-only / implicit gating 没有额外收益，论文核心论点需重估。

---

## H. Codex 本轮只读审查清单（请逐项回应，不准跳过）

### H1. 仓库与实验资产

- [ ] 当前 `git status`、branch、HEAD commit；是否有未提交研究代码，保全方案。
- [ ] `ManiFeel` 所有关联入口：dataset loader、augmentation、normalizer、task sampling、obs encoder、policy backbone、diffusion scheduler、rollout evaluator、config/checkpoints/logging。
- [ ] 历史 Multi-task DP / HyperNet 模块的真实功能、输入 mask 覆盖、tensor shapes；明确哪些只能作参考、哪些能直接复用。
- [ ] 数据与 checkpoint 本地路径是否存在及权限；不公开任何 secret/token，不自动 push。

### H2. 数据与 Label 可行性

- [ ] **逐任务表**列出 3V+2T+P 的真实 keys、各 tensor shape、语言来源与 task instruction。
- [ ] 6D/7D action 的单位/坐标系/控制语义与有效维度 mask；如不能统一则说明是否必须分任务 policy head。
- [ ] 时序对齐与 future tactile lookup 无越界/泄漏。
- [ ] Stage 0 dropout 十种 condition（full + 9 masks）的 mask 实现位置与 indicator。
- [ ] 10 forward-equivalents paired diffusion 共享 `noise/timestep/noisy action/target` 的具体函数改造点。
- [ ] 标签缓存/版本/标准化/每个任务的统计，含负值、过大值处理。

### H3. Stage 1/2 架构与训练

- [ ] `C1` 的输入输出以及学生 Action DP 的 shape/初始化可行性。
- [ ] Future latent encoder 是否可以提供 frozen target；Raw demo action 路径不泄漏；预测范围可实现。
- [ ] Utility/Future/Action 梯度表与可能冲突；损失缩放的审查方案。
- [ ] Stage 2 **冻 E+C1+U** 的实现位置（`eval/train mode`、BN/Dropout/optimizer param groups）。
- [ ] 四个 modality-specific HyperNet 的输入输出/共享规则、FiLM identity initialization。
- [ ] Query 的 `z + utility embedding` 与 `C2` CrossAttn 的维度/位置编码/梯度。
- [ ] `z'_t=z_t` 的 zero-step equivalence **可执行测试**；DP 能否复用 Stage 1 checkpoint。
- [ ] 计算成本估计（7 路 Encoder、Teacher 10 forwards、训练显存、数据 I/O、不同训练阶段时间成本）。

### H4. 论文相关审查

- [ ] 对照 *Regulating Modality Utilization within Multimodal Fusion Networks*：它的 metric/optimization 与本文离线 signed paired-loss labels + prediction head 有何区别？不能把它称作本方案的直接复现。
- [ ] 对照 *MS-Bot*：State/Stage-conditioned Query 的真实机制与本方案 `z + U(u)` 差异。避免只根据项目图臆测公式。
- [ ] 对照 *Representation-Aligned Tactile Grounding*：其 VLA intermediate prediction 的输入、loss 与我们的 `P(z,a_demo)` 不同在哪；提出低成本 DP probe 方案。
- [ ] 对照 HyperTASR / 现有 HyperNet：有哪些本质不同、哪些只是 sensor-level application？若没有明显差异，明确提醒研究创新风险。

---

## I. 交付格式（Codex 必须给出，缺一项视为不完整）

请输出一个**中文 Markdown 审查报告**，建议命名：`docs/ADAVIP_FULL_FRAMEWORK_AUDIT_REPORT.md`（**本轮先在对话中汇报，不直接写仓库**；得到许可后再保存）。

报告按以下顺序组织：

1. **Executive Verdict**：总体是否可实施，分 Stage 0/1/2 给出 `GO / CONDITIONAL / BLOCKED`，不超过 15 行。
2. **Evidence Matrix**：每项重要陈述标注 `[代码已确认] / [数据已确认] / [论文已确认] / [仅设计假设] / [未核实]`，并提供文件路径和行号；无凭据不可写「已确认」。
3. **Full Data Schema Table**：逐任务 7 输入真实字段、shape、missing modality、obs/action/future horizon、action semantics、language source；发现不一致标 P0 blocker。
4. **Model Graph & Tensor Shapes**：Stage 0/1/2 核心前向图，给出每个关键模块输入输出 tensor shape、freeze/unfreeze 与梯度路径。
5. **Top Risks (P0/P1/P2)**：按导致 training collapse、数据泄漏、结果不可信的严重度排序，逐项附证据、最小复现办法和修复方案。
6. **Refactor Plan by File**：`existing path -> change/add -> reason -> expected API -> tests`；清楚区分**小修**和**必须重构**，原则是沿用现有环境/训练器而重做 full-sensor pipeline。
7. **Stage-gated Training Plan**：每阶段训练目标、最低开销 smoke test、必须观察的 curves/metrics、进入下一阶段的定性/定量门槛（阈值可提建议，不能伪装已确定）。
8. **Ablation/Compatibility Plan**：公平 baseline、checkpoint 可移植边界、Stage 2 identity test、对照实验列表。
9. **Open Questions for Human Approval**：按阻断性排序，最多 10 项；每项给一个首选方案及备选方案与利弊。若仓库事实能回答，不得把它作为问题反问用户。
10. **Minimal Next Coding Batch**：只有用户同意后才开始的最小改动（优先全传感器 dataloader/schema + masked simple-fusion DP 的单 batch smoke test），**不要跨 Stage 写完整三阶段模型**。

**审查过程权限约束：**

- 允许只读命令：`git status/log/branch`、查看文件、`rg/grep/find`、读取本地 config 和样本 metadata；如果需读取少量实际 samples，确保不更改数据/缓存。
- 不允许：`git checkout/reset/clean`、新建分支、修改训练代码、启动训练、自动下载大型模型/数据、上传私有内容、`git push`。任何可能写入磁盘的 smoke test 或环境安装先单独申请许可。
- 若源码缺失/数据不可访问/论文结论无法证实，请直接列为 blocker，不得填假 shape 或猜测效果。

---

## J. 推荐 Codex 执行顺序（只读，按风险优先级）

**Pass 1 — Ground Truth of Repository & Dataset**：检查数据结构、所有 7 路输入是否能读取、历史 multi-task 代码的缺口、各任务 action semantics、时间对齐。任何 P0 缺口立即报告。

**Pass 2 — Teacher Feasibility**：检查全传感器 DP/Task conditioning/Dropout/10 paired forwards 需要哪些新增接口，确认可实施且 Teacher 评估可用。

**Pass 3 — Student Training**：核查 `C1+Utility Head+action/future latent` 的梯度/信息泄漏/target encoder 和 checkpoint 初始化策略。

**Pass 4 — Adaptive Fusion**：核查 frozen representations、四类 HyperNet-FiLM、MS-Bot-inspired Query、`C2` residual identity 初始化与与旧 DP 的兼容。

**Pass 5 — Report Only**：提交 I 节规定的报告，先等人审查，不执行重构。

---

## K. 外部核对资料（只提供验证入口；本文未把所有实现细节视为已证实）

1. ManiFeel official paper: https://arxiv.org/html/2505.18472 ；特别是 III-A observations、Appendix Table V（仿真 3 camera/2 tactile/7D proprio；6D/7D action）；Appendix C-D 的 DP/encoder training config。
2. MS-Bot project / source: https://gewu-lab.github.io/MS-Bot/ 、https://github.com/GeWu-Lab/MS-Bot/ （stage-conditioned state/query 与动态 fusion；**不是**我们 9D Utility 的直接先例）。
3. Regulating Modality Utilization within Multimodal Fusion Networks (Sensors 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11435562/ （其目标 utilization 调节与本方案 teacher-derived signed label regression 不同）。
4. Representation-Aligned Tactile Grounding: https://arxiv.org/html/2607.14609 （VLA intermediate representation + future latent tactile grounding；本方案 DP adaptation 待实验验证）。
5. HyperTASR: https://arxiv.org/pdf/2508.18802 （HyperNet 机制参考，非天然 novelty）。
6. 旧讨论交接记录与手绘图：本会话所附 `AdaViP_Research_Discussion_Handoff_2026-10-17(1).md`、`BO1.pdf`。

---

## L. 重要裁决：哪些已定、哪些不能提前决定

**已定（可进入架构审查）**：multi-task full-sensor teacher；7 路 basic inputs；9D loss-difference utility；Stage 0 dropout reference DP；Stage 1 C1 + Action + Utility + future tactile latent；Stage 2 freeze E/C1/U + modality-shared HyperNet-FiLM + utility-conditioned query + residual `C2` 输出 `z'_t`。

**待 Codex/试验确认**：实际 9/10 task 配置、传感器确切 keys、TacRGB/TacFF、language token representation、动作空间是否能可靠统一、DP horizon、model dimensions、dropout probabilities、loss weights、future offsets、teacher denoising objective、各模块参数初始化可用性、Stage 2 是否需要局部解冻及各消融的实际有效性。

**绝不在审查阶段预设**：该方法必定优于 attention/gating、Utility 9 维都稳定有效、未来触觉预测必定提升 success、两个 Transformer 必定胜过一个 Transformer、旧 checkpoint 可以一键复用。

**最后要求 Codex 回答一个问题：** 如果现在必须选择 **最稳健、最小可验证的下一步实现**，应当改哪几个文件、怎样用一批数据证明 `3V+2T+P+L` 都正确进入 Full-sensor Multi-task DP，并怎样避免为后续 Stage 1/2 埋下数据/shape/normalization 问题？请先用证据回答，再等我确认是否开始编码。
