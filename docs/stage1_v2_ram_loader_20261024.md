# Stage 1 v2：无损 RAM DataLoader 优化（2026-10-24）

## 范围与安全边界

- 只优化 CPU RAM 读取和组批。模型、batch32、FP32/TF32、loss/权重、归一化、
  EMA、AdamW、九任务 BalancedTaskSampler 和数据顺序均不更改。
- 正式作业384201/server09和 checkpoint 归档385697保持运行。没有修改正式冻结
  source、config.yaml，没有重启或启动多卡训练。原正式 source fingerprint：
  `80ba8e07a5d17dd72521d8ff7245030bd476e03778f66492ac100606e765dc36`。
- 新实现为显式 opt-in；现有训练 workspace 的 DataLoader 入口和配置不变。
  本文的独立短测不意味着优化已经应用于正式训练。

## 实现

1. `CachedStage1Dataset` 复用已校验的 `SourceIndexedArray`，直接查 compact row_map。
   连续行使用切片；非连续/重复行保留原次序。只对已解码、已验证的缓存省略重复
   finite/min/max 全帧扫描；原始数组/Zarr仍走完整原路径，缺行和 shape/dtype 仍报错。
2. `RamBatchDataset.__getitems__` 在 worker 内直接分配最终 `[B,T,C,H,W]` 共享存储，
   从 NHWC RAM 视图写入。避免每个 sample 的 NCHW 临时副本及后续大张量 stack。
   小型 state/action/mask/identity 仍使用原 sample 逻辑和 default_collate。
3. Stage1 索引构建和运行时检查已保证全部 future 行有效；因此取消 future image
   的 zeros 分配和布尔 scatter，直接写有效四帧。没有改变 padding/mask 语义。
4. 保留原生 DataLoader pin-memory 线程、8 workers、prefetch_factor=2、私有 generator；
   没有在 worker 初始化 CUDA，没有改为半精度/uint8，没有缓存会随 EMA 更新的特征。

新入口：`adavip/manifeel/stage1_ram_loader.py::make_ram_dataloader`，接收
`CachedStage1Dataset` 和原 DataLoader 参数，包括原 sampler/generator。只允许精确的
CachedStage1Dataset 类型，以免绕过子类自定义 `__getitem__` 额外字段；不自动替换
生产 dataset，也不更改生产配置。

## 正确性验证

- 首轮相关测试40 passed：含新增 RAM batching、既有 Stage1、parallel、DDP contract、
  profiler 测试；这里的 DDP 为 CPU 回归测试，不是启动多卡训练。
- 九任务、6/7维 action、两个 episode 的边界、不同 epoch/resume offset、重复/乱序
  索引、最后不完整 batch、0/2 workers，共同验证所有张量、dtype、mask、identity
  与原 DataLoader bitwise 相同，缓存只读且修改输出不会污染源数据。
- 连续/非连续 compact row_map、缺行、错误缓存 shape、非缓存回退有覆盖。
- 最终完整 `pytest -q`：270 passed、1 skipped、1 failed，143.45s。
  唯一失败为既有 `test_film_checkpoint_uses_local_assets` 缺本地 CLIP source，
  不涉及本次 RAM 路径。最终相关测试41 passed，RAM单独测试12 passed。
- 独立 H200 实验还将与正式冻结版 dataset 逐张量比较256个真实窗口，并比较相同
  checkpoint/RNG 下原始与优化路径的训练 loss/梯度范数。结果以实验报告为准。

## 独立 H200 A/B 方法

- 先在CPU上准备256个真实窗口，取原 BalancedTaskSampler(seed42) 的有序前缀，
  覆盖九任务。运行时全部加载到匿名 RAM；不是 mmap 按需读，不在计时区间读盘。
- baseline 动态导入正式冻结版 `stage1_v2_dataset.py`，不是用优化后的基类冒充原版。
  `original_a → cached → batched → original_b` 同节点依次运行，每次独立 Python
  进程，恢复同一个只读固定 checkpoint；模型和 optimizer、精度完全一致。
- 每种预热24步，正式计时160步。计时包含完整 H2D、forward、backward、clip、Adam、
  EMA update 和相同标量日志同步；排除初始化、RAM载入/校验、预热、Checkpoint I/O。
  为隔离 loader，短测固定 checkpoint LR，不推进调度器或执行梯度审计。
- 1×H200、32CPU、128GiB、申请4天，排除server09；不是正式训练的新续跑。
  每秒独立采集 GPU utilization/memory，并记录 CUDA event H2D、loader wait、吞吐。
- 这是约10GB RAM子集循环，而不是完整279GiB随机工作集，不能将短测加速比直接
  等同正式训练39.7samples/s的提升。重复 baseline 用于判断节点漂移。

## 实验结果

### CPU 数据组织微测

同一个256窗口、9.824GiB匿名RAM子集，单个CPU线程，batch32，每路径预热1batch，
随后3轮×8batch；不含 DataLoader worker/pinning/H2D/GPU，不能当训练吞吐。

| 路径 | 平均 CPU 组批时间 | 相对原版 |
|---|---:|---:|
| 正式冻结原版 | 1475.52ms/batch | 1.00× |
| verified cache + 原生 stack | 1018.29ms/batch | 1.45× |
| 直接写最终 batch buffer | 401.17ms/batch | 3.68× |

直接组批耗时降低72.8%；这只证明CPU数据组织更快，不代表正式训练快3.68倍。
原始结果：`cpu_batch_assembly.json`。

### H200 对比

独立作业385896/server34已完成，15分33秒、COMPLETED0:0；初始检查 NVIDIAH200，143771MiB，0MiB已用、
0%GPU利用率、无计算进程。固定参考checkpoint为epoch17/step64000。256窗口
按九任务计数 `[29,29,29,29,28,28,28,28,28]`，原始数组10,548,568,544bytes。

| 顺序/路径 | samples/s | steps/s | 输入等待均值 | H2D均值 | GPU Util均值 |
|---|---:|---:|---:|---:|---:|
| original_a | 33.325 | 1.041 | 431.19ms | 25.54ms | 32.94% |
| cached | 77.437 | 2.420 | 61.04ms | 25.07ms | 76.68% |
| batched | 75.716 | 2.366 | 69.12ms | 25.02ms | 75.82% |
| original_b | 69.820 | 2.182 | 101.86ms | 25.03ms | 73.36% |

全部显存peak allocated34.962GiB、NVML40751–40755MiB。首3个训练step及第184个step
的全部已记录loss、latent variance和梯度范数完全相同（四路径最大绝对差0）；cached
和batched各自256个真实窗口逐张量/字典顺序bitwise校验全部通过。

**重要：原版两次相差2.095倍，存在严重基线漂移。不能宣称2.32倍训练加速，也不能
用两次原版均值计算有因果意义的1.50倍加速。** 相对于更快原版69.820，cached和batched
的观测比为1.109/1.084，但第一轮不足以给出稳定提升幅度。汇总工具在baseline max/min
超过1.10时自动将speedup置null、`speedup_interpretable=false`，不掩盖异常。

直接batch buffer虽然CPU单线程更快，但本轮H200没有比简单cached路径更快，不应据CPU
微测优先启用更复杂的组批实现。追加385927，同资源/精度/固定checkpoint，
复用相同256窗口和冻结case代码，做 `original_a → cached_a → cached_b → original_b`
配对复测；不把第一轮的可疑冷基线当定论。复测结果在 `repeat/` 子目录。
13:13 CST核验：385927已在server05完成，COMPLETED0:0，耗时14分10秒，
`repeat/summary.json`已生成。original_a/b为64.220/64.307samples/s，cached_a/b为
64.905/65.156samples/s；基线漂移仅1.001345倍，cached相对原版均值的观测差约
1.00%/1.39%，没有显示明显端到端提速。两次cached均通过256窗口bitwise校验，
首3步及末步loss/梯度范数最大绝对差为0；仍只是固定子集，不是全数据性能。
另发现报告中original的torch_threads=1、cached=8，比较还受主进程线程数混杂影响；
自动汇总的speedup_interpretable仅检查基线漂移，不足以证明因果加速，后续严格
对照必须统一线程设置。没有新提交测试或迁移正式训练。

实验根目录：
`/data/wangzihao/outputs/manifeel/manifeel_stage1_v2_ram_loader_20261024_122628`。
最近目录指针：`/public/home/wangzihao/.cache/avs1v2/current_ram_loader.txt`。
冻结代码/hash、准备日志、测试输出、逐张量校验、单秒NVML与各case报告均在此。
作业使用提交时脚本快照；随后新增的summary工具不改运行case，完成后在CPU上汇总。

## 正式启用前

本轮不迁移正式训练。新代码改变 source fingerprint；不得覆盖生产 snapshot、改写旧
checkpoint provenance、放宽 resume 校验，或偷偷更换 loader。后续如获准启用，需独立
输出目录、完整模型/optimizer/scheduler/RNG/normalizer/EMA 恢复与 sample_cursor
对齐，明确记录代码/loader版本迁移，保留原 checkpoint 和正式运行。
