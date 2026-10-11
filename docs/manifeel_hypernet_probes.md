# Power Plug HyperResNet 机制 Probe

## 范围

本页记录 CPU 示范特征分析、GPU 动作敏感度和模拟器闭环对照。
第一轮按用户要求只聚焦 Power Plug；原因排查阶段获准只补1–2个任务数据，实际只增加
Peg Insertion 的约252MiB子集。结果入口见 `results/README.md`。

下面的 CPU 部分读取保存的 EMA encoder、Hydra encoder 配置以及 observation normalizer。
RGB 预处理复用上游示范加载器的 OpenCV resize、HWC→CHW 和 checkpoint
normalizer，再经过 encoder 内部的 ImageNet normalization。没有重新拟合 normalizer，
也没有实例化 diffusion action backbone 或 Isaac Gym。完整 checkpoint 在加载时仍会
临时占用 CPU 内存；加载后只保留 encoder 和 observation normalizer。

## CPU 示范特征结果

结果根目录：

`results/hypernet_batch8_20261010/`（仓库根目录下，已加入 `.gitignore`）

每个模型都读取 `plug_quan_Aug02` 的 episode 0、1、2，各均匀采样 64 帧，
合计 192 帧。使用 CPU、2 个 Torch 线程、batch size 8、无 DataLoader workers，
且命令设置 `CUDA_VISIBLE_DEVICES=''`。采样依据是固定 episode 编号，不是模型表现。
这些示范来自训练时使用的数据源，不是独立的测试 episode。

| 模型 | Checkpoint | 结果子目录 |
| --- | --- | --- |
| 主模型，encoder HyperNet + fusion | `checkpoints/0926/dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt` | `power_plug_epoch100_demos012` |
| 额外 fusion HyperNet，同 epoch 对照 | `checkpoints/0929ours/fusionhyper/latest_epoch100.ckpt` | `power_plug_fusionhyper_epoch100_demos012` |
| 额外 fusion HyperNet，已有选定 checkpoint | `checkpoints/0929ours/fusionhyper/latest_epoch300.ckpt` | `power_plug_fusionhyper_epoch300_demos012` |

先看根目录的 `comparison.png`，再看主模型子目录的
`episode1_signals.png`、`episode1_modulation_centered.png`、`episode1_inputs.png`。
主模型共 12 张图；两个额外 fusion HyperNet 模型各 15 张。

每个结果子目录包含：

- `summary.json`：checkpoint、预处理配置、总体及每条示范的描述统计。
- `hypernet_demo_trace.npz`：逐帧原始/调制后特征、γ/β、attention、相对残差、
  数据集索引与示范内帧号；模态轴顺序为 vision、tactile。
- `episode*_signals.png`：调制幅度、相对固定调制的变化、各 head 的 TacRGB attention、
  原始 force-field 信号和 normalized state norm。
- `episode*_modulation.png`：两模态的通道 × 时间 γ/β 热图。
- `episode*_modulation_centered.png`：减去每个通道的示范内均值，突出时间变化；
  各面板使用独立的对称色阶，99% 分位数之外的颜色会截断。
- `episode*_inputs.png`：同一时刻的 wrist/TacRGB 输入；不是 saliency map。
- `episode*_fusion_hypernet.png`：仅额外 fusion HyperNet 模型具有的调制图。

## 指标定义和初步观察

总调制幅度为 `||alpha * (gamma * feature + beta)|| / ||feature||`。
动态部分则在保持当前原始特征不变时，将 γ/β 替换为该示范的通道均值，
计算调制特征差的范数并除以当前原始特征范数。这个均值利用整条已采样示范，
仅用于离线反事实描述，不是可部署策略或动作级干预。

以下均为描述均值，单位是相对原始模态特征范数的百分比：

| 模型 | vision 总调制 | tactile 总调制 | vision 动态部分 | tactile 动态部分 |
| --- | ---: | ---: | ---: | ---: |
| 主模型 epoch 100 | 3.05% | 28.98% | 1.16% | 10.47% |
| + fusion HyperNet epoch 100 | 3.46% | 16.27% | 0.69% | 1.66% |
| + fusion HyperNet epoch 300 | 4.24% | 20.63% | 1.13% | 3.82% |

主模型的四个 head 对 TacRGB 的平均 attention 约为
`[1.000000, 0.999993, 0.000004, 1.000000]`。在这三条示范的采样帧上，
更像三个 head 固定读触觉、一个固定读视觉，而不是随阶段切换模态。
γ/β 则确实存在示范内变化，但尚未对齐物理阶段。
主模型 `alpha=0.01148063`，`tanh(fusion_gate)=-0.03405872`；
attention 概率没有计入 value/output projection、负的 fusion gate、残差路径
或直接传给 policy 的 tactile 特征，不能当作最终模态贡献。

**相对调制幅度不能直接比较成“信任比例”。**例如主模型 vision/tactile
原始特征平均范数约为 11.06/1.09，实际残差平均范数约为 0.338/0.258。
触觉相对百分比更大，一部分来自更小的分母；不能写成“触觉贡献是视觉的十倍”。
额外 fusion HyperNet 自身的相对调制均值约为 3.97%（epoch 100）和
4.86%（epoch 300），但这些内部差异不能直接解释其成功率差异。

## 复现

从仓库根目录运行，选择尚不存在的输出目录：

```bash
source scripts/manifeel/common.sh
activate_manifeel
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$MANIFEEL_PYTHON" scripts/manifeel/probe_hypernet_demos.py \
  --checkpoint checkpoints/0926/dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt \
  --output-dir results/hypernet_batch8_20261010/power_plug_epoch100_demos012_recheck \
  --episodes 0 1 2 --frames-per-episode 64 --batch-size 8 --threads 2
```

额外 fusion HyperNet 的复现只需替换 checkpoint 和输出目录。
可使用 `--dataset` 覆盖保存的数据路径。仅加载可信 checkpoint；Torch/dill
checkpoint 反序列化可能执行代码。

## CPU 特征图的结论边界

- 192 个相邻/采样帧不是 192 次独立实验；只有三条探索性示范，没有显著性结论。
- 总体 γ/β 标准差包含示范之间的差异；判断示范内变化应查看各 episode 的统计和图。
- 不自动按时间三等分或 force-field 阈值标注阶段；触觉接触可能来自夹持物体，
  不等价于与目标首次接触。阶段仍需物理状态或人工核验。
- 图中变化不证明 policy 因此改变动作，也不证明变化导致成功率提升。
- 同 epoch 模型对照也不是仅改变一次推理调制的干预；两者分别训练。
  epoch 100/300 对照还混合了训练时间差异。
- CPU 特征 probe 不计算成功率。GPU 闭环结果使用 `test/mean_score`，
  不用这些内部指标或末尾窗口的 `test/success_rate` 替代。

## GPU 追加结果

用户授权 GPU5，以及空闲的 GPU3/4/6。短 preflight 后完成如下实验，均保持
checkpoint 中的 EMA、normalizer、100 diffusion inference steps；没有启动训练。

### 三模型闭环复现

50 环境、500 steps、seed100000；初始 state 和 plug/socket 位姿完全一致：

| 模型 | GPU | `test/mean_score` | 输出子目录 |
| --- | ---: | ---: | --- |
| baseline epoch100 | 3 | 0.50（25/50） | `gpu3_rollouts/baseline_epoch100_n50_steps500_seed100000` |
| 主模型 epoch100 | 4 | 0.66（33/50） | `gpu4_rollouts/hyperresnet_epoch100_n50_steps500_seed100000` |
| 额外 fusion HyperNet epoch300 | 6 | 0.56（28/50） | `gpu6_rollouts/fusionhyper_epoch300_n50_steps500_seed100000` |

主模型相对 baseline 新成功 11 个、丢失 3 个成功环境。这是已有评测 seed 的复现，
不是独立 seed 复制或因果架构消融。`rollout_report/` 包含汇总图和原始统计。

物理阶段按 plug/socket pairwise force 定义，阈值 `0.001`；首次成功之后不纳入。
阈值 `0.0001/0.001/0.01` 下曾接触环境数不变（baseline44、主模型46、额外模型46）。
以环境为单位平均并 bootstrap，而不是把相邻帧视作独立样本。
主模型接触前/后触觉相对残差约为 0.319/0.220，attention 均接近 `[1,1,0,1]`，
不支持“接触后触觉权重上升”的简单叙事。

### 配对噪声动作敏感度

GPU5 上使用 demos0/1/2，每条 8 个 histories，3 个采样 seeds11/22/33，batch8。
模糊/延迟一个模态，同时保持其余输入和 diffusion 噪声不变。固定 γ/β 的校准
使用另外 demos3/4/5，各 8 个 histories；没有在测试 histories 上拟合均值。
重复控制的归一化动作 RMS 精确为零。

| 干预 | 主模型 | baseline |
| --- | ---: | ---: |
| vision blur | 0.189153 | 0.175710 |
| tactile blur | 0.013986 | 0.026947 |
| vision delay10 | 0.144821 | 0.133755 |
| tactile delay10 | 0.010082 | 0.027323 |
| state delay10 | 0.020589 | 0.020191 |
| 固定 γ/β | 0.009748 | 不适用 |
| 延迟 γ/β | 0.006617 | 不适用 |
| 关闭 γ/β | 0.017523 | 不适用 |

动作差乘以保存的 action normalizer scale 后，在 action chunk 和 6 个维度上取 RMS。
两模型 scale 完全相同。触觉敏感度更低不能区分抗噪与忽视触觉，不能解释成贡献更高。

空间 probe 在 episode1 预选帧33/102/170，对两帧 history 同位置做4×4局部 blur
（kernel31、sigma10）。`action_probe_report/action_occlusion_comparison.png`
统一跨模型和模态色阶；这是动作敏感度，不是 attention 或成功 utility。

### 真实接触前后的调制干预

`gpu5_contact_actions/hyperresnet_epoch100_reconstructed_control/` 选取主模型
rollout 前20个具有接触前后、成功前 histories 的环境，每环境一对；优先与首次接触
相隔16 steps，若无这样的 history 则取可用最近样本。只更改 γ/β，不更改当前
raw features、state、alpha、fusion 参数。

固定 γ/β 的平均动作 RMS 为接触前0.00419、接触后0.00270；关闭为0.01609/0.02324；
前后阶段互换为0.00409/0.00605。详细模态拆分和原始值见结果 README/summary。
这说明动作可能受调制影响，但不测量动作正确性或任务成功 utility。

数值控制：缓存特征重建 encoder condition 最大绝对误差2.98e-7。Attention 重分批
计算会有微小浮点差异，经过100步 diffusion 可放大，所以不能直接把另一批次保存的
condition 当作配对基准。最终采用同批次动态重建 condition 为基准，重复动作严格
逐位一致。先前 `gpu5_contact_actions/hyperresnet_epoch100/` 的失败尝试不参与结论。

### 小样本闭环干预

同一主模型、seed200000、10环境、500steps：原始7/10、固定8/10、关闭8/10。
初始 state/位姿完全一致；两个干预只比原始多成功同一个环境。样本过小，
不能证明干预更好，也不支持动态调制优于静态/关闭。不得与50环境成绩直接相减。
这是推理干预，不是重新训练的架构消融。图与配对环境统计在 `controlled_rollout_report/`。

## GPU 部分复现命令

以下命令是复现示例，必须先重新确认所用 GPU 空闲且已获授权；用新输出目录避免覆盖。
Simulator 使用物理 GPU 索引，不设置 `CUDA_VISIBLE_DEVICES` 重映射；独立 policy
probe 使用 `CUDA_VISIBLE_DEVICES=5` 和逻辑 `cuda:0`。

```bash
source scripts/manifeel/common.sh
activate_manifeel
MAIN=checkpoints/0926/dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt
ROOT=results/hypernet_batch8_20261010
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/manifeel/eval_single_task_checkpoint.py \
  --checkpoint "$MAIN" --device cuda:4 --n-test 50 --n-test-vis 2 \
  --max-steps 500 --test-start-seed 100000 --hypernet-trace \
  --output-dir "$ROOT/recheck_main_n50"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/manifeel/probe_action_sensitivity.py --checkpoint "$MAIN" \
  --device cuda:0 --episodes 0 1 2 --samples-per-episode 8 --seeds 11 22 33 \
  --calibration-episodes 3 4 5 --calibration-samples 8 --batch-size 8 \
  --output-dir "$ROOT/recheck_action_sensitivity"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/manifeel/probe_action_occlusion.py --checkpoint "$MAIN" \
  --device cuda:0 --episode 1 --steps 33 102 170 --grid-size 4 --batch-size 8 \
  --output-dir "$ROOT/recheck_action_occlusion"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/manifeel/probe_contact_condition_actions.py --checkpoint "$MAIN" \
  --run "$ROOT/gpu4_rollouts/hyperresnet_epoch100_n50_steps500_seed100000" \
  --calibration-trace "$ROOT/gpu5_action_probes/hyperresnet_epoch100_demos012/action_sensitivity.npz" \
  --max-environments 20 --batch-size 8 --output-dir "$ROOT/recheck_contact_actions"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/manifeel/eval_single_task_checkpoint.py \
  --checkpoint "$MAIN" --device cuda:3 --n-test 10 --n-test-vis 2 \
  --max-steps 500 --test-start-seed 200000 --hypernet-trace \
  --hypernet-intervention mean \
  --modulation-trace "$ROOT/gpu5_action_probes/hyperresnet_epoch100_demos012/action_sensitivity.npz" \
  --output-dir "$ROOT/recheck_mean_n10"

CUDA_VISIBLE_DEVICES='' python scripts/manifeel/compare_power_plug_action_probes.py \
  --results-root "$ROOT" --output-dir "$ROOT/recheck_action_report"
```

Baseline checkpoint 为 `checkpoints/batch8_tactile/baseline/latest_epoch100.ckpt`；
其动作 probe 只选 `--variants repeat vision_blur tactile_blur vision_delay tactile_delay state_delay`。
额外 fusion HyperNet 为 `checkpoints/0929ours/fusionhyper/latest_epoch300.ckpt`。
闭环动态/关闭对照分别使用 `--hypernet-intervention none/zero`，匹配其他参数。

## 第一轮尚缺的机制证据

1. 多 seed、更大样本的动态/静态/关闭闭环干预，量化成功率差异的不确定性。
2. 仅接触前或接触后启用干预，建立阶段特异的 utility，而不是动作变化。
3. 匹配训练协议和 checkpoint selection 的重新训练消融，在 `engram` 上进行。

当前不应继续把结果表述成“已经证明 policy 在不同阶段相信不同模态”。
以下追加排查补齐其中一部分；第一轮10环境表不是最终机制样本量。

## 追加原因排查：选择混杂、数值饱和和 cross-conditioning

完整图表、指标、复现命令和 GPU 运行状态见
`results/mechanism_followup_batch8_20261010/README.md`。本阶段没有训练；用户确认后
已启动两个新 seed 的50环境闭环干预和预先选定 checkpoint 的独立 seed 复核。

- 既有50环境、seed100000结果中，baseline100/200/300为50/64/58%，主模型为66/66/62%；
  同epoch差16/2/4pp，三者取最高只差2pp，但在旧评测seed选最高仍有选择偏差。
  HyperNet无fusion为60/66/70%，静态fusion为52/54/58%，不应假定fusion解释全部收益。
- 抽查保存配置确认seed42、batch8、50 demos、val_ratio0.02、100 diffusion inference
  steps；normalizer逐元素相同。ResNet权重不同只证明训练表示不同，不证明差异有益。
- Power Plug成功前1563个inputs的真实Q/K logits重放误差3.96e-9；所有head的max
  probability均≥0.99，logit差中位数约17/15/−14/17。Attention固定有直接数值饱和证据。
  模型仍有触觉直达policy路径，不能称为没用触觉。
- 主模型γ/β约79–92%的分量处于abs≥0.95区间，alpha0.01148；仍有环境内变化，
  不是完全常量。当前架构是残差特征调制，不是显式模态信任gate。
- 同时刻、同接触阶段、未成功环境之间只交换HyperNet context，保持真实features/state
  不变。触觉context引起的fused condition相对变化为接触前0.525%、接触后1.314%；
  是值得进一步验证的cross-conditioning候选，尚非阶段utility，且扰动幅度未跨阶段标准化。
- 11个主模型100新增成功中，baseline100有8个始终离目标超过2倍成功距离阈值、
  3个曾接近；丢失的3个成功在主模型中均曾接近。此为几何描述，不是因果阶段归因。
- 新增Peg的96个示范采样inputs也存在attention饱和，但几乎全部选择视觉；其数据来自
  训练来源，没有新算Peg闭环成绩。下载后只保留6条完整示范和5个keys，压缩包/分片已清理。

### 应如何解释新增闭环干预

已完成全程100环境结果：动态66%、固定均值62%、关闭68%。固定减动态−4pp，
条件配对环境bootstrap 95%区间[−9,0]pp；关闭减动态+2pp，区间[−3,+7]pp。
未校正配对sign test p分别0.21875/0.6875。没有证明关闭更好或等价，也没有建立动态
调制的必要性。原始 trace、恒定/全零核验与配对初始状态检查保存在
`results/mechanism_followup_batch8_20261010/controlled_matrix_report/summary.json`。

- 原始、全程示范均值、全程关闭、仅接触前均值、仅接触后且成功前均值都使用同一
  主模型 EMA，不是重新训练。均值来自独立于早期动作 probe 的示范3/4/5，但仍来自
  训练来源数据，不能声称是 held-out demonstration generalization。
- 两个新 evaluation seeds300000/400000，每个50环境、500步、100 diffusion inference
  steps。逐环境验证初始位姿、奖励最大值以及实际 γ/β；阶段 mask 从已执行的
  plug/socket pairwise force 和 reward 历史重建，阈值0.001，排除首次成功后的决策。
- 区间是固定 evaluation seeds 内的配对环境 bootstrap；这里只有一个训练 seed42。
  不包括训练方差，也没有做等效性检验；“未发现显著损失”不等于“两个方法完全等价”。
- 分阶段干预改变的是决策策略，不是独立可相加的收益分解；接触前的改变会影响后续
  状态分布，不能简单用 pre/post 分数差作为某阶段贡献百分比。
- 全程关闭若不损失成绩，最多说明当前样本未证明**推理时**调制必要；训练中的
  表征/协同优化作用仍是候选解释，不能由此宣称 HyperNet 在训练中没有用。

### 下一台训练服务器上的最小区分实验（本机不执行）

1. 保持 batch8、数据/观测/normalizer、扩散步数、优化器等现有协议。至少多个配对训练
   seeds；预注册统一 epoch 或只用验证集选 checkpoint，不在 test seed 上反复挑最高。
2. 比较 baseline、无 fusion HyperNet、主模型、**可学习但不依赖 context 的 γ/β**。
   加上保存主模型的推理均值/关闭对照，区分表示学习、静态通道重标定和真正动态 utility。
3. 仅在需要检验 attention 饱和候选原因时，比对 Q/K normalization 或受控 temperature，
   保持其他路径不变；记录 projected logits、熵、各分支 norm 和梯度，不只画 attention。
4. 额外 fusion HyperNet 的较低成绩不能由现有跨 checkpoint 对比唯一解释。
   对该**同一 checkpoint**单独冻结/关闭额外 branch，随后再做匹配训练消融，才有机会
   区分推理时干扰与训练协同/选择差异；不能假定“参数更多所以过拟合”已经被证明。

本轮没有启动这些训练，也没有擅自改变原九任务 benchmark 表。

## 最终补齐结果：阶段干预与独立checkpoint复核

有效新评估共16组，每组50环境/500步，最多两个同时运行；比较复用100个匹配初始条件，
不是800个独立初始环境。两evaluation seeds的初始条件确实不同；所有配对初始状态逐元素
一致。阶段mask与已执行contact/reward历史一致，干预选择的γ/β确实恒定，非阶段标签后验拟合。

| 同一主模型的推理策略 | seed300000 | seed400000 | 合计 |
| --- | ---: | ---: | ---: |
| 原始动态 | 72% | 60% | 66% |
| 全程示范均值 | 70% | 54% | 62% |
| 全程关闭γ/β | 74% | 62% | 68% |
| 仅接触前固定 | 70% | 62% | 66% |
| 仅接触后、成功前固定 | 74% | 64% | 69% |

仅接触前固定减原始0pp，条件配对95%区间[−7,+7]pp；仅接触后固定+3pp，区间[−3,+9]pp。
都不支持动态调制在对应阶段具有已证明的必要性，也没有做等效性检验。

预先按旧seed选择的模型在两个新seeds合计：主模型100=66%、baseline200=54%、
HyperNet无fusion300=64%、静态fusion300=59%。主模型对baseline+12pp，条件配对95%
区间[+2,+22]pp；新增成功21、丢失9，未多重校正探索性sign test p=0.04277。
只有一个训练seed且保存epoch不同，不能据此唯一隔离架构原因或声称总体训练显著性。
旧seedbest-of-three仅差2pp并不意味着独立新seed上提升消失。

新100环境的已接触但始终几何不到位失败从27减到15；21个新增成功中15个来自这类失败。
这是对齐/到位几何的描述性线索，**不是接触前动态调制或接触后触觉精修的因果证据**。
训练时表示学习/协同优化、静态调制与收敛速度差异更值得下一台服务器验证，当前仍未证实。

图表总览及PDF：`results/mechanism_followup_batch8_20261010/evidence_overview/`；
完整数值/复现/失败尝试边界见其上级README。原9任务canonical成绩保持不变。
本机没有训练，只新增一个252MiB Peg子集；本次GPU进程已全部退出，其他人的进程未终止。

内存故障处理是评估实现改动，不是模型改动：释放载入后无用的训练workspace/payload，
将并发限制为两个，并只回收本次读取checkpoint的clean file pages；没有改图像预处理、
模型权重、normalizer、diffusion sampler或成功判定。停滞/中止尝试不计入成绩。
