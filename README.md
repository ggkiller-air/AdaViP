# AdaViP ManiFeel Simulation

本仓库只保留 AdaViP 在 ManiFeel Isaac Gym/TacSL 中的仿真训练与评估流程。
真机策略、数据处理和部署代码由独立仓库维护；运行这里的实验不需要 ROS、
相机、机器人 SDK 或真机数据。

## ManiFeel 仿真

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

### 6. 单任务数据和官方 DP smoke

只做环境验证不需要数据。要跑官方视觉 DP，先下载目标 Zarr 数据集，例如
USB insertion：

```bash
export MANIFEEL_DATA_ROOT=/path/to/datasets/manifeel
bash scripts/manifeel/download_dataset.sh usb_quan_Aug05.zip
export MANIFEEL_DATASET_PATH="${MANIFEEL_DATA_ROOT}/usb_quan_Aug05"
```

然后执行一个极短的单任务训练 smoke（2 epochs、每 epoch 2 个 train step、1 个
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

当前主入口是单任务训练：

```bash
bash scripts/manifeel/train_single_task_dp.sh
```

默认使用 USB 的 `vision_wrist` 配置；可通过
`MANIFEEL_TASK_CONFIG`、`MANIFEEL_ISAACGYM_CONFIG` 和
`MANIFEEL_DATASET_PATH` 切换到其他单任务/触觉配置。multi-task 训练脚本和
已有九任务结果仍保留用于历史复核，但不属于当前主实验入口，也不需要为本
阶段重新训练九个任务。

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

## 其他目录

- `adavip/model/`、`adavip/policy/`：AdaViP 模型与策略组件。
- `adavip/manifeel/`：ManiFeel 数据、runner、workspace 和 observation encoder。
- `configs/manifeel/`：Power Plug 单任务配置和一份历史 multi-task DP 配置。
- `scripts/manifeel/`、`slurm/manifeel/`：本地与集群训练、评估入口。
- `third_party/`：固定版本的 ManiFeel、Isaac Gym fork 与 Diffusion Policy。

`third_party/diffusion_policy` 是 ManiFeel 原版 DP 和当前 Power Plug 训练的
直接依赖，不是可以移除的历史副本。TacSL Isaac Gym 授权文件不会纳入 Git。

## 开发

CPU 单元测试：

```bash
PYTHONPATH="$PWD:$PWD/third_party/diffusion_policy" \
  pytest -q tests
```

提交数据、checkpoint、日志和生成文件前请检查 `.gitignore`。这些产物应保存在
仓库外部。

## 许可证

本仓库由 Zihao Wang 维护，并采用 [Apache License 2.0](LICENSE)。
`third_party/` 中的项目分别遵循其上游许可证。
