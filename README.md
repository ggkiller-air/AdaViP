# AdaViP

AdaViP 是面向视觉、触觉与本体感知的机器人策略训练仓库。仓库当前包含
真机 Reactive Diffusion Policy（RDP）训练流程和 ManiFeel 仿真实验；两套
环境相互独立。

> 真机协作者只需阅读下面的“真机 RDP 训练”，不需要安装 Isaac Gym、
> ManiFeel、ROS、机器人 SDK 或部署端依赖。

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
