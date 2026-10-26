# 在 4090 上评测 Stage2 Transformer / MLP

`main` 同时包含 `Stage2Policy`（Transformer）和 `Stage2MlpPolicy`，统一使用
`eval_full_sensor_checkpoint.sh`。从 ckpt 的 fusion 参数自动识别模型，默认严格加载
EMA 和训练时的 normalizer；不需要 Teacher、训练数据、Utility 标签或 RAM cache。
旧 Stage0 ckpt 继续使用同一个入口。

## 更新代码和准备文件

在 4090 的评测目录更新（不要在正在训练的目录切分支）：

```bash
git switch main
git pull --ff-only origin main
git submodule update --init --recursive
```

沿用 4090 上已配置的 ManiFeel / Isaac Gym 环境，按需设置
`MANIFEEL_ENV_PREFIX`、`MANIFEEL_PYTHON`、`MANIFEEL_ISAACGYM_ROOT` 和缓存根目录；
参见 README 的环境设置。代码更新不会安装 CUDA、Isaac Gym 或下载模型文件。

本批 ckpt 位于 ModelScope `ggkiller/Manifeel-ckpts1018` 的
`stage2_t2_jobs387293_387295_20261026/`，保留如下结构：

```text
stage2_t2_jobs387293_387295_20261026/
  manifest.json
  transformer/checkpoints/epoch{20,50,100,200,300}.ckpt
  mlp/checkpoints/epoch{20,50,100,200,300}.ckpt
```

另外保留**训练时冻结的** `task_embeddings.npz`（Stage0 E500 发布包中同一份
九任务 CLIP ViT-B/32 embedding），通过 `--task-embedding-path` 传入本机路径。
不要根据任务名称重新生成另一套 embedding。

旧 Stage2 ckpt 没有保存 Hydra 配置。仓库中的
`configs/manifeel/eval_stage2_t2.yaml` 提取自两次训练共用的冻结 Teacher E500，
仅保存推理架构和观测规格：T=2、horizon=16、8 步动作、100 步 DDPM、
三路 256×256 相机、两路 320×240 触觉。权重、RMS 和 normalizer 全部来自
当前 ckpt。其他架构用 `--config /path/to/inference.yaml` 显式提供配置。

## 单个 checkpoint

```bash
bash scripts/manifeel/eval_full_sensor_checkpoint.sh \
  --checkpoint /path/to/transformer/checkpoints/epoch300.ckpt \
  --task-embedding-path /path/to/task_embeddings.npz \
  --output-dir /path/to/eval/transformer_epoch300 \
  --device cuda:0 --n-test 50 --max-steps 500
```

换成 `mlp/checkpoints/epoch300.ckpt` 即可评测 MLP，无需切分支或指定模型类型。
默认逐任务启动独立进程，输出每个任务的 `test/mean_score` 和九任务
`test/macro_mean_score`；`--task-id power_plug_insertion` 可仅测一个任务。
每次使用新的输出目录，避免覆盖已有结果。

## 批量评测两种模型

```bash
bash scripts/manifeel/eval_stage2_checkpoints.sh \
  --checkpoint-root /path/to/stage2_t2_jobs387293_387295_20261026 \
  --task-embedding-path /path/to/task_embeddings.npz \
  --output-dir /path/to/eval/stage2_comparison \
  --epochs 20 50 100 200 300 \
  --device cuda:0 --n-test 50 --max-steps 500
```

顺序评测 Transformer 和 MLP，按模型/epoch/任务保存结果和独立日志，每完成一个
ckpt 更新 `summary.json`。缺少文件时在启动前报错，单次评测失败时停止并保留已有
结果。用 `--variants mlp` 或更短的 `--epochs` 列表评测指定子集。

## CPU 检查与 RTX 相机检查

只检查参数和 normalizer 的严格加载，不启动 GPU 或 Isaac Gym：

```bash
bash scripts/manifeel/eval_full_sensor_checkpoint.sh \
  --checkpoint /path/to/mlp/checkpoints/epoch300.ckpt \
  --output-dir /path/to/check/mlp_epoch300 --check-only
```

批量入口也支持 `--check-only`。CPU 检查不能证明 RTX 仿真可运行。
正式 rollout 前，在 4090 上对实际使用的 GPU 做有超时的相机预检并记录物理索引
和 UUID。激活 `common.sh` 中的环境后可运行：

```bash
source scripts/manifeel/common.sh
activate_manifeel
nvidia-smi --query-gpu=index,uuid,name --format=csv
timeout 120s "$MANIFEEL_PYTHON" scripts/manifeel/diagnose_isaacgym_camera.py \
  --access gpu --frames 100 --num-envs 1 \
  --compute-device-id 0 --graphics-device-id 0 > /path/to/camera_preflight.log 2>&1
rg diagnostic_complete /path/to/camera_preflight.log
```

要求命令正常退出并出现 `diagnostic_complete`；失败或超时后先排查该卡的图形链路。
预检通过后，先用 `--task-id power_plug_insertion --n-test 2 --max-steps 10`
做完整策略/触觉通路短测，再跑正式协议。本次整合服务器未启动 GPU 仿真。
