# Full-sensor Multi-task DP：Stage 0 基础实现

## 2026-10-18 Checkpoint 对比评估

当前单卡长训仍使用300epoch预算，未切换到1000epoch；不要将下文DDP扩展能力
当作已运行的实验。训练继续进行，评估使用独立冻结副本，不读取写入中的`.partial`。

ModelScope仓库`ggkiller/Manifeel-ckpts1018`的
`full_sensor_stage0_batch64_20261018/`内包含：

- `checkpoints/latest.ckpt`：冻结时已完成276epoch，step299000，下一batch368。
- `checkpoints/epoch18.ckpt`：完整epoch18，step19476，原训练目录的`best.ckpt`。
- `task_embeddings.npz`、`training_config.yaml`和含大小/SHA256的`manifest.json`。

两份checkpoint均保存raw/EMA权重、训练normalizer和optimizer。入口默认只恢复
EMA推理模型，不加载训练数据、不重新拟合normalizer，也不分配optimizer。
默认full input，注入窗口内恒定的语言embedding，按任务截断6D/7D动作；缺少
任意训练传感器或历史/分辨率不匹配会报错，不静默降级。左右TacRGB保持320×240。

下载后，在已配置IsaacGym的评估机器上先做小规模检查：

```bash
ARTIFACT_ROOT=/data/wangzihao/checkpoints/full_sensor_stage0_batch64_20261018
bash scripts/manifeel/eval_full_sensor_checkpoint.sh \
  --checkpoint "$ARTIFACT_ROOT/checkpoints/epoch18.ckpt" \
  --task-embedding-path "$ARTIFACT_ROOT/task_embeddings.npz" \
  --output-dir /data/wangzihao/outputs/manifeel/stage0_epoch18_smoke \
  --device cuda:0 --task-id power_plug_insertion --n-test 2 --max-steps 16
```

正式对比对两份checkpoint分别使用不同输出目录，去掉`--task-id`，设置
`--n-test 50 --max-steps 500`；保持默认`--test-start-seed 100000`和
`--policy-seed 42`一致。九任务各自独立进程，逐任务输出`test/mean_score`，
总表输出`test/macro_mean_score`，不以旧`test/success_rate`作正式结论。
按需设置`--n-test-vis`保留视频；小规模smoke不视为正式benchmark结果。
当前已完成CPU回归及真实checkpoint恢复检查，尚未宣称此入口通过GPU仿真smoke。

上传脚本`upload_full_sensor_checkpoints_modelscope.sh`使用ModelScope上传API，
显式unset全部代理变量，不使用Git传checkpoint；先复制完整文件再上传，保留
上传时真实计数和SHA256，不覆盖正在训练的latest。

## 同节点四卡 DDP 与 1000 轮续训

入口 `slurm/manifeel/train_full_sensor_ddp_batch64.sbatch` 使用一节点四张 H200、
48 CPU、512 GiB 主存。配置中的 `dataloader.batch_size=64` 始终表示**全局**
batch，每卡正常批次 16；LR 保持 `8e-5`，不做线性放大，不减少每轮更新数。
69,205 次 task-balanced 有放回采样仍对应 1,082 次 optimizer/EMA 更新。
最后 21 个样本按 `[6,5,5,5]` 分片，用 `world_size * local_sum / global_count`
反传，抵消 DDP 梯度均值，得到真正的全局逐样本均值。更小尾批的空 rank 只做
零权重 dummy forward，不增加有效样本，也不使用 DistributedSampler 补齐。

使用原有约 278.93 GiB float32 NPY 缓存的只读 mmap；同节点各 rank/worker
映射相同 inode，共享 OS page cache，而非各自加载一份 NumPy RAM。rank0 首先
SHA256 校验和预热，其他 rank 重新校验并附着；记录 NPY 映射的 RSS/PSS 和
文件身份。spawn DataLoader worker 只序列化文件路径/行索引，重开只读映射，
不会 pickle 全部图像，也不在已初始化 NCCL 的进程中 fork worker。

训练每批聚合全局 task/condition sums/counts。验证对原始顺序无重复分片，对
各 task/condition 的 sum/count 做 all-reduce，再计算 micro/macro；每轮覆盖
全部九任务 2,060 样本。验证噪声按原始全局 batch 生成后切片，同一样本十个
条件配对，使用 EMA。各 rank 每个全局更新仅更新一次 EMA，并在初始化和完整
epoch 后核对 raw/EMA 全量权重 SHA256 和 update cursor。

只允许 rank0 原子写 checkpoint，仍保存裸 policy、EMA、optimizer、Hydra cfg、
train-only normalizer、采样 cursor、全局统计，以及**每 rank** Python/NumPy/
CPU Torch/当前 CUDA RNG。DDP 原样恢复要求 world size 不变。单卡迁入保留已有
权重、optimizer、EMA 和 cursor，其他 rank 获得独立、可复现的后续 RNG。
固定 DDP 梯度桶布局以避免重新构造 reducer 改变浮点归约；CPU/Gloo 回归检查
加载时权重/optimizer 逐位一致，以及续训后与不中断训练逐位一致。不能据此
承诺不同 GPU 算法/拓扑之间也逐位一致。第三方的零元素 device anchor 在本地
policy 中冻结，不改 `third_party/`。

短测保留原 300 轮 cosine schedule，运行至两个 epoch 边界（至少一轮完整），
再真正退出并重启进程，从 checkpoint 多训练 120 updates；使用 runtime stop，
**不通过缩短训练预算伪造短测 LR**。`assess_full_sensor_ddp.py` 同时检查共享
缓存、采样、EMA、验证、恢复与两段实际吞吐；相对仍在运行的单卡最近 20 个
窗口要求至少 1.15x 稳态训练吞吐，且数据等待不超过 35%；另外匹配相同 epoch，
包含 worker 启动的完整训练吞吐也必须至少 1.15x。另记录包含验证和
保存的完整 epoch wall time；稳态吞吐倍数不是端到端墙钟加速倍数。任何失败
都不停止原单卡，也不自动升格为长训。

通过短测后才从现有单卡**最终安全 checkpoint**切换到 1000 轮（不是拿较早
短测分支覆盖最新训练进度）。1000 轮总预算为 1,082,000 次更新；DDP 本身不
改变 steps/epoch。预算延长须显式 `runtime.allow_epoch_extension=true`，其余
科学配置仍严格一致；以 checkpoint 当前 LR 为锚连续衰减至新终点，不重置
warmup，也不把过去的 300 轮 cosine 当成从一开始就使用了 1000 轮。

长训采用 `ADAVIP_MODE=long ADAVIP_EPOCHS=1000 ADAVIP_STOP_EPOCHS=0
ADAVIP_ALLOW_EXTENSION=true`，并要求 `ADAVIP_VALIDATION_REPORT` 指向已通过
的短测报告；通过 sbatch 覆盖为 48h/最短 6h backfill。批处理 USR1 提前 300s
写本地 stop-file，rank 在安全边界保存；只有这种限时安全退出才自动提交同
source/config 的 afterok 续训。普通 TERM/失败不自动重启。

Checkpoint 策略：早期 `epoch1/2/5/10/20/50.ckpt`（仅尚未经过的轮次），之后
`epoch100/200/.../1000.ckpt` 永久保留；每轮和每 1,000 updates 滚动覆盖
`latest.ckpt`，全输入 EMA macro validation loss 改善时更新 `best.ckpt`，
中断另存 `interrupt.ckpt`。已过早期轮次不伪造历史权重，迁移输入 checkpoint
单独保留供审计。所有训练产物位于 Git 外的持久数据根。

## 范围与入口

本实现独立于旧的多任务 DP、HyperResNet 和 FiLM 实验，不修改
`third_party/`，也不将旧 checkpoint 当作 full-sensor teacher。

- 配置：`configs/manifeel/train_full_sensor_reference.yaml`。
- 数据：`adavip/manifeel/full_sensor_dataset.py`。
- 编码与融合：`adavip/manifeel/full_sensor_obs_encoder.py`。
- DP / paired loss：`adavip/manifeel/reference_diffusion_policy.py`。
- 训练 / checkpoint：`adavip/manifeel/full_sensor_workspace.py`。
- 契约：`adavip/manifeel/full_sensor_schema.py`，schema `adavip.full_sensor.v1`。

当前只优化 action denoising loss。不实现 C1、Utility Head、Future Predictor、
HyperNet、C2 或 utility cache。future tactile API 仅提供索引/有效性边界。

## 七路输入与模型图

固定顺序为 `front, side, wrist, left_tactile_camera_taxim,
right_tactile_camera_taxim, state, task_embedding`，对应 `V1,V2,V3,T1,T2,P,L`。

默认 Camera 为 `[B,2,3,256,256]`，左右 TacRGB 为 `[B,2,3,320,240]`。
`state` 是 `[B,2,7]` 的末端位置与 xyzw quaternion，不是关节状态。
语言使用已有冻结 CLIP archive 的每任务固定 512D embedding，在窗口内重复。
没有将 `task_index` 送入网络，也没有使用 per-task action/state normalizer。

五个独立、从头训练的 ResNet18 使用 GroupNorm；每路两帧特征拼接后投影到
256D。P 的两帧拼接投影，L 投影一次，产生 `[B,7,256]`。传感器 identity
保持固定。各路不发生交互前，用 learned missing token 替换被删除的 token，
然后拼接七路 token 和七个 presence flags，经轻量 MLP 输出 `[B,256]`。
U-Net 使用这个**窗口级**条件，而不是误用 `[B*To,d]` 编码再展平。

图像由 `[0,1]` 统一正规化为 `[-1,1]` 一次，不叠加 ImageNet normalization。
没有随机 crop/augmentation；编码和 mask 都可在 eval 模式确定性复用。
Stage 0 的简单融合不等价于未来 C1；不得假定其权重可无条件移植。

## Full / Dropout 分布

condition 固定顺序为：

| condition | 删除输入 | 默认概率 |
| --- | --- | --- |
| full | 无 | 0.8 |
| V1 / V2 / V3 | 对应单路 Camera | 各 0.2/9 |
| T1 / T2 | 左 / 右 TacRGB | 各 0.2/9 |
| P / L | Proprio / Language | 各 0.2/9 |
| vision | 全部三路 Camera | 0.2/9 |
| tactile | 全部两路 TacRGB | 0.2/9 |

每个样本独立抽取一个 condition，同一样本整段历史共用 mask，不组合多个
removal。80/20 是采样期望，不是每个 batch 的硬配额。
`policy.full_input_probability` 控制 full 比例；`policy.mask_weights` 的九个
非负权重控制剩余概率，顺序与表一致。默认九个权重均为 1。
`policy.full_input_probability=1.0` 是 no-dropout 对照。

## 轨迹、时序与 action 契约

默认 `H=16, To=2, Ta=8`。每个真实 timestep `t` 生成一个样本：

- obs 索引为 `t-To+1 ... t`。
- action 索引为 `t-To+1 ... t-To+H`。
- 执行 `action_pred[:, To-1:To-1+Ta]`，第一个执行动作对应当前 `t`。
- 只在本 episode 内重复边界，不跨 episode；重复 action 的 time validity 为 0。
- `action_loss_mask=[H,7]` 是 time validity 与任务有效 action dimensions 的乘积。
- 6D 任务第七维不参与拟合统计或监督；7D 任务保留真实 gripper target。
- loss 对每个样本先除以自身有效元素数，再对样本求平均。
- 无效 action 槽在 normalized 空间置零后再加 diffusion noise；不会让重复值
  或 6D padding 的非零归一化偏移污染有效预测。

原始 action 的前三维是 world delta 命令，缩放 0.01 m；旋转为左乘的
axis-angle 增量，缩放 0.05 rad；第七维为 finger joint position target m。
没有把 rotation 改为 quaternion，也不直接裁剪可能超出 `[-1,1]` 的 raw action。

`observation_valid_mask` 与 `action_valid_mask` 随样本返回。重复 obs 边界与部署
初始化历史兼容，仍作为输入；其 mask 是 provenance，不是额外的 sensor removal。
`get_future_tactile(index, offsets)` 以当前 `t` 为 anchor，使用正 offset，限制到
`H-To`；episode 末尾越界 target 置零并返回 false validity，不送入当前 obs。

数据中未验证采样频率，`source_frequency_hz=null`。不将 frame offset 自动换算成
10 Hz 的秒数，不宣称当前仿真控制频率等于演示频率。

## Split 与 normalization

每任务按完整 episode 用 `seed + task_index` 划分，至少一条 validation 和一条
train episode。`max_train_episodes` 只下采样 train，其余未选轨迹不会成为 validation。
重复 source store 会被拒绝，避免两个任务配置对同一轨迹各自 split 导致泄漏。

`get_normalizer()` 始终只读取选中的训练轨迹，validation dataset 调用同样规则。
state 拟合训练帧；action 各列独立拟合，仅用该列真实存在的任务帧；image 为固定
range mapping；language identity。统计冻结并完整进入 raw/EMA checkpoint。
全 6D 数据中的缺省 gripper 使用有限的零统计，不用 padding 拟合。

本地九任务检查：400 episodes、71,265 frames；默认 split 为 391 train episodes /
69,205 train frames 和 9 validation episodes / 2,060 validation frames。
这只对应本地九个 source stores，不代表论文所有配置已经复现。

## Paired denoising 接口

```python
policy.eval()
result = policy.paired_denoising_losses(batch, generator=generator)
absolute = result["losses"]   # [B,10], full then nine removals
utility = result["utility"]  # [B,9], masked loss - full loss
```

一次调用只 encode observations 一次，构造一次 `DenoisingSample`，十个 condition
复用同一 noise、timestep、noisy action、target 和 action/time mask。支持 epsilon /
sample prediction，保留 raw signed differences，不将负值裁成零。
要求 policy、encoder 和 denoiser 全部 eval；不生成 rollout 或真实成功率标签。
可以显式传入 `noise` / `timesteps` 做可复现对照。

`compute_loss_per_sample(...)` 返回 `[B]`，`compute_loss(...)` 是其均值。
`prepare_denoising_sample(...)` 加 `denoising_loss_per_sample(...)` 是底层共享接口；
准备好的 sample 只能用于其来源 batch，不可跨 batch 复用。

未来生成正式 utility 标签必须另行选择、冻结 teacher checkpoint，记录其 hash、
版本、split、normalizer、样本索引及采样随机性。当前训练中的 validation 不等于
对外发布的冻结 teacher 标签集，也未确定重复噪声采样次数 K。

## 训练、验证与断点

默认 train/val 都是 batch8，FP32、AdamW、warmup + cosine LR、EMA、grad clip。
H/To/Ta、U-Net 默认宽度和基础优化参数沿用旧 DP 设置；任务均衡采样、七路独立
编码、融合、mask、严格 split/normalizer 和 checkpoint 校验是本次实现方案，
不是官方最优超参数声明。每 epoch 以 `len(train_dataset)` 为采样预算，对任务赋
等概率质量，有放回采样，不要求每个 epoch 遍历每条 train frame。

验证对 full 和九种 mask 使用**同一个 EMA**，不用 raw-model loss 选 EMA teacher。
保存各 task / condition 的 per-sample 加权均值、micro / macro loss、训练 condition
覆盖数、grad norm、吞吐率和 CUDA 峰值 allocated memory。best 以 full-input
macro validation loss 选取；只有 loss 验证，没有仿真 success 验证。

`latest.ckpt`、`best.ckpt`、周期 checkpoint 同步原子写入，保留 Hydra-compatible
config、raw/EMA、normalizer、optimizer、epoch/global step、EMA step 和 RNG。
resume 保存 epoch 内下一个 batch 的 cursor，重放相同的任务均衡 sampler 并直接跳过
已用索引，不重读已消费帧；同时保留部分 epoch 的统计。CPU 回归验证中途/边界
中断恢复和连续训练逐位一致。
不保证不同硬件或非确定 CUDA kernel 之间逐位相同。

manifest 保存数据路径、shape/dtype、episode ends、train/val IDs、action codec、
embedding archive SHA256 与窗口定义；provenance 记录 manifest、normalizer 和训练
协议指纹。resume 拒绝数据/split/统计/协议变更，包括修改 dropout、batch、epoch
预算或 schedule。可改 output_dir、resume 路径，以及经过无损校验的 decoded cache
路径/模式和 DataLoader workers、prefetch、pinning 等传输参数；不是自动追加 epoch
的续训工具。旧版完整协议哈希仅在 checkpoint 自带配置验证通过、排除上述运行时字段
后的科学协议完全一致时迁移，不绕过数据或 normalizer 校验。
已有 latest 的 output_dir 不允许无 resume 覆写。checkpoint 仅加载可信来源。

**边界：** manifest 不对巨大图像存储逐块哈希，不能检测保持 metadata/统计不变的
原地数据替换。source stores 必须只读、不可变；更换内容时必须更新 dataset_version。
不支持分布式训练或混合精度；本阶段刻意优先简单可验证路径。

### Batch64 初次训练与监控（2026-10-17）

用户批准先投递一个训练任务、再依据实测调整 batch。新增 launcher
`slurm/manifeel/train_full_sensor_reference_batch64.sbatch`，正式目标300 epochs，
train/validation batch64，固定80/20 dropout，LR8e-5、warmup500 steps、EMA，
不裁剪 epoch 预算或 validation 数据。本次单卡H200、16CPU、64GiB，首作业限6小时。
`engram` 分区拒绝当前 yangyc 账户，投递使用既有 H200 训练脚本的 `gpu` 分区；
不是在无可见GPU的交互节点 server37 上启动训练。

每20步打印/追加 `progress.jsonl`，记录窗口 samples/s、data-wait fraction、
loss、LR、grad norm、全部mask累计计数与显存；GPU状态每60秒单独写日志。
第一份 latest 在step100保存，随后每1000 steps及每epoch更新；best按完整九任务
EMA full macro validation loss选取；固定epoch快照仍每20轮保存。
计时窗口不包含checkpoint写盘；首窗口含初始化开销，不作稳态估计。

CLI处理USR1/TERM/INT，安全步边界写latest和interrupt后退出；验证中断时保留
训练cursor，恢复后重跑完整验证。Slurm使用 `srun` 转发提前300秒的USR1。
主动安全停止可用 `scancel --signal=USR1 JOB_ID`，避免依赖强制kill前的短暂宽限。
恢复需要原始冻结源码、相同配置和 `ADAVIP_RESUME=/.../checkpoints/latest.ckpt`。
batch变化会触发协议校验拒绝，不能静默修改一个正在恢复的实验。

训练源码复制到run目录的source并记录SHA256和Git/submodule元信息，依赖指向
原始只读third_party；后续开发不改变已投递任务的Python源码。未自动commit/push。

**IPC环境保护：** 过长的源码工作目录不能作为multiprocessing临时目录。launcher
显式使用短路径 `/data/wangzihao/t/mf_JOBID`，先检查临时目录未静默回退、Unix
socket及两worker张量传输，再构建模型；检查总限90秒。train/val多进程loader均
设置180秒timeout，单进程自动置0。作业367952曾因此在首批数据前卡住，已取消；
修复后替换作业368013；原367952没有可用checkpoint，不属于batch64 OOM或收敛失败。

### 无损预解码与长训

原始 Zarr 是 float32/Blosc 压缩存储，随机双帧窗口会重复读块、解压。新的 CPU
入口 `scripts/manifeel/build_full_sensor_cache.py` 一次性解码选定 train+validation
episodes，按任务/传感器写连续 float32 NPY，不量化、不压缩、不缓存训练特征。
只提前执行与原 loader 相同的确定性 resize；未使用的 episodes 不进入缓存。
训练/验证划分、train-only normalizer、窗口、mask、采样预算和监督完全不变。

缓存以源 manifest、Zarr array metadata 和明确 dataset_version 绑定，逐文件保留
shape/dtype/SHA256，原子发布完整 manifest。构建可重入，已完成文件先校验再复用；
未完成文件不会作为有效缓存加载。源 store 仍要求只读不可变；不能以修改时间检查
替代完整源内容哈希，原地换数据必须更新 dataset_version 并使用新缓存目录。

`dataset.decoded_cache_dir=/...` 开启缓存，`decoded_cache_mode=ram` 在主进程一次
以8路并行加载并校验，Linux fork workers 共享只读数组；`mmap` 可用于低内存检查，但不保证
远程 NAS 的随机读取性能。九任务当前所选 71,265 帧缓存共约278.93 GiB，RAM 模式
需额外至少32 GiB作业内存余量。不要使用 spawn workers 复制整个缓存。

构建脚本 `slurm/manifeel/build_full_sensor_cache.sbatch` 只申请 CPU，无需占 GPU
等待解码。长训脚本 `slurm/manifeel/train_full_sensor_decoded_batch64.sbatch` 申请
1H200、24CPU、384GiB host RAM、最多48小时，允许至少6小时backfill窗口并安全接续，
保留batch64/300epochs/完整九任务验证。
运行时先做5个20-step窗口的吞吐预检，丢弃第一个热身窗口；稳态需达到150 samples/s
且 data wait不超过35%，否则保存checkpoint并失败退出，避免低效运行48小时。
通过后在同一进程直接继续，不重复加载279GiB缓存。

长训的临时 IPC 使用节点本地 `/dev/shm/avip_JOBID`，避免 NAS Unix socket 在
worker 退出时产生 `.nfs* Device or resource busy` 清理异常；这些只是短生命周期
socket/控制文件，不存模型或数据缓存。持久缓存、日志和 checkpoint 仍在 `/data`。

沿用每1000步/每epoch latest、每epoch验证best、每20epoch快照。只在时限预警
SIGUSR1触发的安全保存后自动提交依赖当前作业成功退出的续训；失败预检、异常和
普通用户取消不自动重启。最终epoch300正常完成也不再提交。续训仍校验同一完整
科学协议，并恢复optimizer、EMA、RNG和步级cursor。

## 操作与验收

```bash
source scripts/manifeel/common.sh
activate_manifeel
"${MANIFEEL_PYTHON}" scripts/manifeel/check_full_sensor_dataset.py \
    --output /data/wangzihao/outputs/manifeel/full_sensor_stage0/dataset_check.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${MANIFEEL_PYTHON}" -m pytest -q \
    tests/test_full_sensor_stage0.py \
    --basetemp=/data/wangzihao/pytest-tmp/full_sensor_stage0

mkdir -p /data/wangzihao/outputs/manifeel/slurm
sbatch slurm/manifeel/preflight_full_sensor_stage0.sbatch
```

Slurm 预检申请一张 GPU，启动时打印型号/显存/利用率/进程、CPU affinity，使用
真实图像尺寸、五个 ResNet18、默认 U-Net 和 EMA；batch8 两个 optimizer steps，
一批十条件验证，保存后重新加载 checkpoint。限制为 30 分钟，不是正式训练。
一批 validation 只覆盖开头任务，必须看 `validation/task_coverage`，不得把它当作
九任务性能验收。任务逐项数据窗口与七分支梯度另有 CPU 回归覆盖。

2026-10-17 的 H200 预检作业 `367844` 已成功退出：两步训练、十条件验证与
checkpoint 重载通过，峰值 allocated 显存约 8.13 GiB。16 个训练样本的随机
condition 恰好全为 full，不能据此声称 GPU 训练已覆盖九种 removal；显式九种
mask backward 由 CPU 回归验证。详见当日开发日志；这不是 teacher 性能验收。

正式训练入口（仅在批准的训练资源上执行；batch64作业使用上述专用launcher）：

```bash
bash scripts/manifeel/train_full_sensor_reference.sh \
    output_dir=/data/wangzihao/outputs/manifeel/full_sensor_reference_batch8_seed42
```

需要进一步用冻结 checkpoint 跑逐任务仿真，明确左右 TacRGB 为 `[320,240]`，
注入同一 embedding 并按任务截断 6D/7D action。旧 eval 配置不能不经适配直接套用
到本配置；未来正式汇总使用 `test/mean_score`，不能用旧 info-based
`test/success_rate` 取代。没有 rollout 结果之前不能宣称 teacher 达标、正式
九任务 baseline 完成或允许直接进入 Stage 1。
