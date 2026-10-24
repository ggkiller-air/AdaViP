# 新版 ManiFeel Stage 1：单 H200 性能诊断（2026-10-24）

## 结论

1. 正式训练的明显损失在 **RAM 数据组织 / DataLoader / pinned-memory 供应链**，
   不是图像解码：最近一小时实际吞吐 **39.36 samples/s**，抽样日志平均输入等待
   **313 ms/step**，已记录训练段 **477 ms/step**。这两个均值来自每20步的日志采样，
   并非完整 profiler 分区；墙钟吞吐还包含 checkpoint、梯度审计等开销。
2. 6帧已经合并为 `[B*6,C,H,W]` 并行编码，没有逐帧 Python 前向循环。
   5个传感器各用一个独立 ResNet；batch32时每个编码器处理192张在线图像。
3. Future EMA 的4帧也合并为 `[B*4,C,H,W]`，在 `no_grad` 下编码。
   独立实测视觉 **27.4 ms**、触觉 **21.3 ms**，合计 **48.8 ms/step**。
   这是重要成本，但不是最大的单项。
4. **不能直接认为 BF16 会加速这套代码。** 此次同节点、同数据、同起点短测未见加速；
   trace显示卷积进入BF16，但GroupNorm仍是FP32，带来大量精度转换。
   不建议未经进一步优化验证就切换正式训练精度。

## 安全边界与实验范围

- 正式作业384201/server09始终运行；没有停止、重启、改配置、换精度或改源码。
  原始冻结 `stage1_v2_*.py` 和训练YAML的SHA256复核全部一致。
- 独立作业 **385774/server26，1×H200，32CPU，96GiB RAM**；申请4天，
  实际 **6分19秒，COMPLETED 0:0**。提交时显式排除server09，没有启动多卡任务。
  GPU启动检查：H200，143771MiB总显存，0MiB已用，0%利用率，无计算进程。
- 先在CPU作业中准备32个真实窗口，覆盖九任务，数量 `[4,4,4,4,4,3,3,3,3]`；
  对应约1.345GB原始float32数组全部加载到匿名RAM。没有在计时区间进行文件读取或解码。
  使用旧缓存的源元数据/全覆盖约束和所选值检查，并保存子集SHA256；没有重新扫描279GiB全缓存。
- 只读固定原始step62000完整checkpoint作为起点。每种条件恢复相同模型和优化器状态，
  使用checkpoint中的LR及辅助权重；短测不推进正式调度器，也不写正式checkpoint。
- batch32，8个DataLoader workers，原始数据读取/校验/整理方法和原生pin-memory线程；
  worker在CUDA初始化前fork。未启用数据快读补丁，未改变训练采样配置。
- 每种条件预热10步，计时80步；组件测量另取20步，trace另取3步。
  初始化、warmup、checkpoint加载及trace采集不计入吞吐。
  BF16另做200步有限性检查。这不是2000-step或全数据长跑基准。
- **重要限制：固定32窗口的1.345GB工作集，比正式279GiB随机RAM访问更有利于局部性。**
  不同节点的绝对速率也不能直接等同。固定fixture结果不能宣称正式训练已从39升至79 samples/s。

## 正式训练与GPU状态

- 最近15/30/60分钟墙钟吞吐分别 **39.48 / 39.12 / 39.36 samples/s**。
- 60分钟抽样均值：DataLoader等待312.94ms，训练段477.32ms。
  等待约占两段抽样时间之和40%；这里的等待不是已经单独测出的H2D时间。
- 原有每120秒GPU采样：最近一小时30点，利用率均值 **36.0%**、范围0–93%，
  已用显存稳定 **40799MiB ≈39.8GiB**。采样稀疏，不能用作精确GPU忙碌时间积分。
- 有限只读现场检查确认同一H200进程4067609，40748MiB进程显存，无swap。
  主进程RSS约303GiB；CPU允许列表22–37,118–133均属NUMA node0，
  但进程内存约54649MiB位于node1（总量约310344MiB的17.6%）。
  NUMA远端访问是候选因素，尚无对照实验可以归因其具体影响。

## FP32 / TF32 与BF16

冻结训练源码没有autocast或GradScaler；模型主参数为FP32。
同一冻结代码/依赖环境中的独立GPU进程实测：

- PyTorch2.4.1+cu121，cuDNN90100；BF16支持检查通过。
- `matmul.allow_tf32=False`、`cudnn.allow_tf32=True`、float32 matmul precision=`highest`。
- cuDNN deterministic=True、benchmark=False，与正式设置一致。
- FP32 trace确实包含 `f32f32_tf32f32` 卷积kernel，因此FP32权重不意味着所有卷积都禁用TF32。
  正式进程没有打印这些全局flag；这里是同源码同环境测量，不是向正式进程注入检查代码。
- profiling实际报告主线程ATen线程数1；环境OMP/MKL设8。不能只依据环境变量认定运行时线程数。

| 独立测量 | samples/s | steps/s | GPU利用率均值 | CUDA峰值allocated |
|---|---:|---:|---:|---:|
| FP32，输入已在GPU | 89.74 | 2.804 | 80.3% | 34.96GiB |
| FP32，原始RAM DataLoader+H2D | 78.91 | 2.466 | 76.8% | 36.21GiB |
| BF16 autocast，输入已在GPU | 86.46 | 2.702 | 78.6% | 40.20GiB |
| BF16 autocast，原始RAM DataLoader+H2D | 77.38 | 2.418 | 74.5% | 41.45GiB |

利用率来自各计时区间内1秒NVML采样（29–33点），不是SM occupancy。
峰值allocated是PyTorch口径，不含保留但未使用的缓存；NVML显存另见原始CSV及JSON。
BF16/FP32速度比：resident **0.963×**、RAM loader **0.981×**；短测没有加速证据，
不能据80步单次测试声称精确的长期减速百分比。

### 数值检查

- 相同权重、输入、模态mask和随机噪声：Action相对差 **0.1395%**，
  Future Vision **0.0558%**，Future Tactile **0.0522%**。
- 全部可用梯度元素比较：余弦 **0.999259**，梯度范数比 **0.991497**；连接关系一致。
- BF16连续200步loss和梯度范数均有限；所有主参数保持FP32。
  这只支持短期数值可运行，不证明长程收敛、真实数据泛化或模拟器成功率不变。

### BF16为什么没有直接提速

3-step trace中，卷积输入由float变为BFloat16，但 `aten::native_group_norm` 输入仍是float。
`aten::_to_copy` 调用由 **9次增至4941次**（均为3步总数）。
按kernel名字粗分类、排除GPU user annotations后：

- 矩阵/卷积类kernel约 **98.7→53.2ms/step**；确有计算收益。
- 逐元素/转换类kernel约 **62.1→114.2ms/step**；基本抵消收益。
- 全部kernel累计约 **281.2→282.5ms/step**，而非明显下降。
- NCHW/NHWC布局转换也有约 **60.7ms/step（FP32）**，值得独立验证内存布局优化。

这些分类是kernel-name启发式，不是完备算子分区。
原始算子表混合含GPU annotation范围，不可直接将所有Self CUDA百分比相加；
组件比较以独立CUDA event测量为准。

## 每step耗时分解

以下为另行采集的20步CUDA-event区间均值；有hook及事件开销，**不是正式作业实测分项**。
区间可能包括GPU时间线上等待CPU继续发射kernel的空隙，不应称为纯kernel busy time。
GPU输入固定fixture；这一组DataLoader等待接近0.5ms，不能替代正式313ms。

| 区间 | FP32 ms | BF16 ms |
|---|---:|---:|
| CPU→GPU实际传输区间 | 24.99 | 24.60 |
| Forward总区间 | 136.97 | 157.63 |
| 其中：5路在线ResNet | 71.30 | 81.88 |
| 其中：Temporal Encoder | 1.69 | 1.52 |
| 其中：Diffusion U-Net | 8.11 | 10.51 |
| 其中：Future EMA视觉3路 | 27.41 | 31.65 |
| 其中：Future EMA触觉2路 | 21.35 | 24.34 |
| Backward全模型 | 177.03 | 180.93 |
| Gradient clip + AdamW | 39.59 | 48.56 |
| EMA参数更新 | 8.27 | 10.85 |
| 指标同步 | 0.27 | 0.27 |

Forward子项已经包含在Forward总区间内，不可重复相加。
FP32各顶层GPU区间合计约387ms，Future target编码约占13%，Backward约46%。
每batch传输 **1,345,236,160 bytes ≈1.253GiB**，约25ms，相当于53.8GB/s；
CPU enqueue均值仅0.74ms，不能把enqueue时间当作DMA完成时间。
训练每步只调用一次U-Net；100 diffusion steps是推理设置，并非每个训练step做100次U-Net。

## 输入路径的直接代码证据与优先级

1. `SourceIndexedArray.__getitem__` 先以高级索引复制数组；随后 `_read_images`
   再高级索引复制，逐次扫描isfinite/min/max，再做HWC→CHW连续化复制。
   Future分支还先分配零数组再写入结果；随后还有collate/shared-memory、pinning和H2D。
   **预解码、RAM驻留并不消除这些CPU内存操作。**
2. CPU-only、逐值相等的快读对照，320次image-window读取均值
   **2.720→0.892ms（约3.05×）**。这是读图微基准，不是全训练3倍加速；未应用于正式代码。
3. 优先在后续独立对照中验证减少冗余扫描/复制、batch整理与pinning供应链，
   再看NUMA/CPU绑核和输入预取重叠；本次不更改任何正式配置。
4. 模型端优先关注CNN/GroupNorm及其反传、布局转换；Temporal和U-Net前向不是大头。
   BF16需要先解决转换和布局开销，不能仅套autocast就预期加速。
5. EMA future target每步随权重更新，不能永久缓存学习特征而保持原算法。
   不建议为省49ms直接取消target编码或改变EMA更新频率。

## 文件与复现

原始实验根目录：
`/data/wangzihao/outputs/manifeel/manifeel_stage1_v2_single_h200_profile_20261024_115517`

- `source/`：实际执行的独立脚本与Slurm脚本，`source_sha256.txt`绑定版本。
- `reference.ckpt`：只读使用的固定完整训练快照；正式latest的原子替换不改变该inode。
- `fixture.json`、`arrays/`：窗口身份、任务覆盖、原始合同来源和所选值SHA256。
- `production_analysis.json`：正式训练日志的15/30/60分钟统计。
- `job_385774/report.json`：原始测量；`report_with_telemetry.json`另附NVML窗口统计和trace分析。
- `job_385774/components_*.json`、`throughput_*.json`、`bf16_numeric.json`：分项原始数据。
- `job_385774/trace_{fp32,bf16}.json`：可用Chrome tracing/Perfetto打开的3步compute trace；
  trace本身使用GPU-resident输入，DataLoader/H2D依据另行的墙钟和CUDA-event测量。
- `gpu_385774.csv`：独立GPU的1秒采样；Slurm日志在公共`outputs/manifeel/slurm/`目录。
- CPU回归入口：`pytest -q tests/test_stage1_profile.py`，6项通过。
- 另一次全量回归在72通过、1跳过后等待既有filelock，未完成；等待约7分钟后
  仅中止本次CPU pytest，不删除锁、不干预持锁者或正式训练。不能将其报告为全量通过。

准备与提交使用 `scripts/manifeel/profile_stage1_v2.py --prepare`，
以及 `slurm/manifeel/profile_stage1_v2_single_h200.sbatch`；提交环境包含
`ADAVIP_SOURCE_ROOT=<正式冻结source>`、`ADAVIP_PROFILE_ROOT=<独立实验根目录>`，
并传入 `--account=yangyc --exclude=server09`。复现须使用新输出根目录，不能覆盖本次结果。
