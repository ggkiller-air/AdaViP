# AdaViP

AdaViP 是面向视觉、触觉与本体感知的机器人策略训练仓库。仓库当前包含
真机 Reactive Diffusion Policy（RDP）训练流程和 ManiFeel 仿真实验；两套
环境相互独立。

> 真机协作者只需阅读下面的“真机 RDP 训练”，不需要安装 Isaac Gym、
> ManiFeel、ROS、机器人 SDK 或部署端依赖。

## ManiFeel 仿真（独立流程）

这一节只面向 Isaac Gym/TacSL 仿真。它与下面的真机 RDP 流程相互独立：
不需要 ROS、相机、机器人 SDK、真机数据或 Piper 部署环境。另一台服务器
只想运行 ManiFeel 仿真时，按本节配置即可。

### 1. 代码和目录

ManiFeel 通过三个 submodule 提供仿真、TacSL Isaac Gym fork 和 Diffusion
Policy：

```bash
git clone https://github.com/ggkiller-air/AdaViP.git
cd AdaViP
git submodule update --init --recursive \
  third_party/manifeel \
  third_party/manifeel-isaacgymenvs \
  third_party/diffusion_policy
```

默认路径可以直接使用；如果服务器路径不同，在运行脚本前覆盖这些变量：

```bash
export MANIFEEL_ENV_PREFIX=/path/to/envs/manifeel
export MANIFEEL_DATA_ROOT=/path/to/datasets/manifeel
export MANIFEEL_CHECKPOINT_ROOT=/path/to/checkpoints/manifeel
export MANIFEEL_OUTPUT_ROOT=/path/to/outputs/manifeel
```

脚本默认使用 Python `${MANIFEEL_ENV_PREFIX}/bin/python`、`cuda:0`，并把输出
写到 `${MANIFEEL_OUTPUT_ROOT}`。不要把数据、checkpoint、Isaac Gym 压缩包或
仿真输出提交进 Git。

### 2. Isaac Gym/TacSL 授权包

ManiFeel 不能使用普通 Isaac Gym 包，必须使用 TacSL 专用的授权归档。请从
[ManiFeel 官方 README](https://github.com/ggkiller-air/manifeel) 提供的
Google Drive 链接下载 `IsaacGym_Preview_TacSL_Package.tar.gz`，放到：

```text
third_party/IsaacGym_Preview_TacSL_Package.tar.gz
```

`scripts/manifeel/setup_environment.sh` 会自动解压到：

```text
third_party/IsaacGym_Preview_TacSL_Package/isaacgym/python/
```

这个归档受许可证约束，不能上传到 GitHub，也不要用普通 `isaacgym` wheel
替代。若放在其他位置，设置 `MANIFEEL_ISAACGYM_ARCHIVE` 和
`MANIFEEL_ISAACGYM_ROOT` 指向实际路径。

### 3. 安装和环境检查

安装脚本创建 Python 3.8 的独立 Conda 环境，并安装已验证的 PyTorch/CUDA、
TacSL Isaac Gym、ManiFeel、Diffusion Policy 依赖：

```bash
bash scripts/manifeel/setup_environment.sh
bash scripts/manifeel/check_environment.sh
```

`setup_environment.sh` 需要 Conda 或 Mamba；首次安装建议在有 GPU 的节点上
执行。检查结果至少应包含 `import isaacgym: present`、
`import isaacgymenvs: present`、`import manifeel: present`，以及
`cuda available True`。登录节点没有 GPU 时，环境安装仍可完成，但 CUDA
检查要在 GPU 节点重新运行。

如果服务器的 CUDA 驱动或镜像访问有问题，可在执行前调整：

```bash
export MANIFEEL_UNSET_PROXY=1       # 代理导致下载失败时
export MANIFEEL_HF_ENDPOINT=https://hf-mirror.com
```

不要把 `requirements-real.txt` 当成 ManiFeel 环境依赖；它是下面真机 RDP
流程使用的另一套 Python 环境。

### 4. 先做无数据 headless 仿真 smoke

`smoke_env.sh` 只创建一个 USB 仿真环境，执行 5 个 reset/step，不加载训练
数据或 checkpoint，适合首先验证 Isaac Gym、CUDA graphics interop 和 TacSL：

```bash
bash scripts/manifeel/smoke_env.sh
```

成功时末尾应出现：

```text
official headless environment reset/step: OK
```

有 Slurm 的服务器可以直接提交仓库脚本：

```bash
sbatch --export=ALL,MANIFEEL_ENV_PREFIX=/path/to/envs/manifeel \
  slurm/manifeel/smoke_env.sbatch
```

脚本默认申请 1 张 GPU、8 CPU、32G 内存和 30 分钟。运行时保持
`MANIFEEL_DEVICE=cuda:0`；Slurm 暴露多张卡时，`cuda:0` 仍表示当前进程的
第一个可见逻辑设备，不要直接填物理 GPU 编号。

### 5. GPU 相机预检（推荐）

如果 smoke 环境通过，再运行 100 帧的最小 GPU 相机检查。它能把 Isaac Gym
相机初始化或 graphics interop 问题与策略/数据问题分开：

```bash
source scripts/manifeel/common.sh
activate_manifeel
export CUDA_DEVICE_ORDER=PCI_BUS_ID
timeout 180s "${MANIFEEL_PYTHON}" \
  scripts/manifeel/diagnose_isaacgym_camera.py \
  --access gpu --image-type color --num-envs 10 --frames 100 \
  --compute-device-id 0 --graphics-device-id 0
```

末尾应出现 `diagnostic_complete`。超时、CUDA native crash 或没有该标记时，
先更换 GPU 节点/物理卡并保留日志，不要马上修改 checkpoint 或重装整个环境；
ManiFeel 的 TacSL 相机故障可能与具体 GPU/节点有关。

### 6. 数据和官方 DP smoke

只做环境验证不需要数据。要跑官方视觉 DP，先下载目标 Zarr 数据集，例如
USB insertion：

```bash
export MANIFEEL_DATA_ROOT=/path/to/datasets/manifeel
bash scripts/manifeel/download_dataset.sh usb_quan_Aug05.zip
export MANIFEEL_DATASET_PATH="${MANIFEEL_DATA_ROOT}/usb_quan_Aug05"
```

然后执行一个极短的训练 smoke（2 epochs、每 epoch 2 个 train step、1 个
validation step、1 个仿真 rollout）：

```bash
bash scripts/manifeel/smoke_train_dp.sh
```

成功 checkpoint 默认位于：

```text
${MANIFEEL_OUTPUT_ROOT}/dp_usb_vision_wrist_smoke/checkpoints/latest_epoch0.ckpt
```

可用同一环境立即做 5-step evaluation：

```bash
bash scripts/manifeel/smoke_eval_dp.sh \
  "${MANIFEEL_OUTPUT_ROOT}/dp_usb_vision_wrist_smoke/checkpoints/latest_epoch0.ckpt" \
  "${MANIFEEL_OUTPUT_ROOT}/dp_usb_vision_wrist_smoke/eval"
```

完整多任务训练才需要下载全部九个 ManiFeel 数据集；可运行
`bash scripts/manifeel/download_all_datasets.sh`，再用
`scripts/manifeel/train_multitask_dp.sh`。长训练和正式评估建议通过对应的
`slurm/manifeel/` sbatch 脚本提交，并先完成本节的 headless smoke 和相机预检。

训练完成后，如需把 multi-task FM 的 epoch 100-600 checkpoint 上传到
ModelScope，可在已登录 ModelScope 的机器上运行：

```bash
bash scripts/manifeel/upload_multifm_modelscope.sh
```

脚本默认读取
`/data/wangzihao/outputs/manifeel/table1_fm_b416_w12_e700_retrain_seed42`，
上传到 `ggkiller/multi-fm`，并自动 unset HTTP/SOCKS 代理。路径或仓库不同
时使用 `MULTIFM_RUN_ROOT`、`MODELSCOPE_REPO_ID` 和 `MODELSCOPE_REVISION` 覆盖；
ModelScope 仓库必须事先创建，且当前账号需要有写权限。

### 7. 仿真故障排查清单

- `isaacgym` 导入失败：确认使用 TacSL 专用归档，并检查
  `MANIFEEL_ISAACGYM_ROOT/isaacgym/python` 是否存在。
- CUDA 不可用：不要在登录/CPU 节点判断仿真失败，申请 GPU 后重新运行
  `check_environment.sh` 和 `smoke_env.sh`。
- 找不到数据：确认 `${MANIFEEL_DATA_ROOT}/<dataset>` 下存在 Zarr 根目录和
  `.zgroup`；环境 smoke 本身不依赖数据。
- 输出或 checkpoint 混乱：每次运行设置唯一的 `MANIFEEL_RUN_NAME` 或
  `MANIFEEL_OUTPUT_ROOT`，不要复用正在运行的 Hydra 输出目录。
- 相机初始化超时或 native crash：保存 Slurm `.out/.err`、节点名、
  `CUDA_VISIBLE_DEVICES` 和 `nvidia-smi` 信息，先换 GPU/节点再做 A/B。

## 真机 RDP 训练

本仓库只负责离线训练。数据采集、Zarr 转换和机器人部署在其他机器完成：

```text
采集/转换机器 -> RDP Zarr -> 本仓库训练 AT 和 LDP -> checkpoint -> 部署机器
```

当前配置对应单臂 Piper、18.75 Hz、7 维关节/夹爪绝对动作。策略使用两个
RGB 视角、7 维关节状态和 15 维 GelSight marker PCA 特征。

### 1. 克隆代码

RDP 以 Git submodule 固定版本。只初始化 RDP 子模块，不需要下载 ManiFeel：

```bash
git clone https://github.com/ggkiller-air/AdaViP.git
cd AdaViP
git submodule update --init --recursive third_party/reactive_diffusion_policy
```

### 2. 安装最小训练环境

推荐 Ubuntu、Python 3.10 和 NVIDIA GPU。以下环境只包含离线 RDP 训练依赖：

```bash
conda create -n adavip-rdp python=3.10 -y
conda activate adavip-rdp

# 下面是已验证版本；CUDA wheel 必须与训练机驱动兼容。
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements-real.txt
```

确认安装和本地集成：

```bash
python -m pytest -q tests/test_real_world_training.py tests/test_piper_rdp.py
```

这条测试不加载 ManiFeel。不要用上游 RDP 的完整 `requirements.txt` 代替
`requirements-real.txt`，前者还包含 ROS、相机、遥操作和部署依赖。

### 3. 准备 Zarr

训练端接收已经转换完成的目录。目录中必须包含：

```text
<dataset>/
└── replay_buffer.zarr/
    ├── data/
    │   ├── action                                  [N, 7]
    │   ├── external_img                            [N, 240, 320, 3]
    │   ├── right_wrist_img                         [N, 240, 320, 3]
    │   ├── right_robot_qpos                        [N, 7]
    │   └── right_gelsight_marker_offset_emb        [N, 15]
    └── meta/
        └── episode_ends                            [E]
```

约定如下：

- 两个图像数组为 `uint8`、HWC 排列。
- 其余数组为有限数值；所有 `data` 数组第一维长度相同。
- `action` 是 6 个关节加夹爪的 7 维绝对目标。
- `episode_ends` 严格递增，最后一个值等于总帧数 `N`。
- 15 维触觉特征必须使用部署端保存的同一组 PCA 参数生成。

先验证数据，不会启动 GPU 训练：

```bash
python scripts/real_world/validate_piper_rdp_dataset.py /path/to/dataset
```

如果转换端已经按上述契约直接生成 Zarr，不需要运行仓库中的数据转换脚本。

### 4. 训练 AT

设置数据和输出目录：

```bash
export ADAVIP_RDP_DATASET=/path/to/dataset
export ADAVIP_RDP_OUTPUT=/path/to/output/seed_42
```

先打印完整命令检查路径和 Hydra 配置：

```bash
python scripts/real_world/train_piper_rdp.py \
  --stage at \
  --dataset "$ADAVIP_RDP_DATASET" \
  --output "$ADAVIP_RDP_OUTPUT" \
  --wandb-mode disabled \
  --dry-run
```

正式训练使用已验证参数：

```bash
python scripts/real_world/train_piper_rdp.py \
  --stage at \
  --dataset "$ADAVIP_RDP_DATASET" \
  --output "$ADAVIP_RDP_OUTPUT" \
  --wandb-mode disabled \
  dataloader.batch_size=256 \
  dataloader.num_workers=8 \
  val_dataloader.num_workers=0 \
  val_dataloader.persistent_workers=false \
  training.num_epochs=601 \
  training.checkpoint_every=10
```

AT checkpoint 默认写入：

```text
$ADAVIP_RDP_OUTPUT/at/checkpoints/latest.ckpt
```

### 5. 训练 LDP

AT 完成后训练 latent diffusion policy：

```bash
python scripts/real_world/train_piper_rdp.py \
  --stage ldp \
  --dataset "$ADAVIP_RDP_DATASET" \
  --output "$ADAVIP_RDP_OUTPUT" \
  --at-checkpoint "$ADAVIP_RDP_OUTPUT/at/checkpoints/latest.ckpt" \
  --wandb-mode disabled \
  dataloader.batch_size=32 \
  dataloader.num_workers=8 \
  val_dataloader.num_workers=0 \
  val_dataloader.persistent_workers=false \
  training.num_epochs=401 \
  training.checkpoint_every=10 \
  training.sample_every=10000
```

最终训练 checkpoint 位于：

```text
$ADAVIP_RDP_OUTPUT/ldp/checkpoints/latest.ckpt
```

显存不足时先减小 `dataloader.batch_size`；GPU 等待数据时再增加训练机 CPU
数量和 `dataloader.num_workers`。W&B 默认关闭，需要时把
`--wandb-mode disabled` 改为 `offline` 或 `online`。

### 6. 自定义数据字段

若新数据不是上述 Piper 契约，需要同时修改：

- `configs/real_world/piper_rdp/task/` 中 AT 与 LDP 的 `shape_meta`；
- Zarr 字段名、图像尺寸、状态维度和动作维度；
- 触觉扩展观测 `extended_obs`；
- 控制频率变化时的 horizon 与时间下采样参数。

不要只修改 README 中的路径后直接训练不同结构的数据。

## 其他目录

- `adavip/`：AdaViP 模型、策略和真机训练适配器。
- `configs/real_world/`：真机训练配置。
- `scripts/real_world/`：真机数据检查与训练入口。
- `configs/manifeel/`、`scripts/manifeel/`：可选 ManiFeel 仿真流程。
- `third_party/`：固定版本的上游依赖，不在本仓库内直接修改。

ManiFeel 的安装与 GPU 仿真说明见 `scripts/manifeel/`，它不是真机 RDP
训练的前置条件。

## 开发

CPU 单元测试：

```bash
PYTHONPATH="$PWD:$PWD/third_party/diffusion_policy:$PWD/third_party/reactive_diffusion_policy" \
  pytest -q tests
```

提交数据、checkpoint、日志和生成文件前请检查 `.gitignore`。这些产物应保存在
仓库外部。

## 许可证

本仓库由 Zihao Wang 维护，并采用 [Apache License 2.0](LICENSE)。
`third_party/` 中的项目分别遵循其上游许可证。
