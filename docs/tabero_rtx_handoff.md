# Tabero RTX rollout 交接（2026-11-02）

目标：在同一台3090服务器完成模型推理和Tabero仿真，验证单环境显存以及
完整closed-loop rollout。用户电脑不参与推理、仿真或转发；运行不依赖H200。
不要训练或修改baseline，不覆盖已有其他benchmark环境。

优先使用同一台服务器的两张空闲3090：一张跑Field+FS策略server，另一张跑
Isaac仿真和渲染，进程只通过127.0.0.1:18018通信。两张卡的显存分别验收。
如只有一张空闲3090，再按同卡方案测试模型与单环境的总显存。两种方案均待
实测，不能用分卡通过来声称同卡24GB通过。

## 固定来源

- 环境：https://github.com/NathanWu7/Tabero
  commit `f57f147c4abd0c223186878ff595e4a0089c0fbe`
- 模型：https://github.com/NathanWu7/Tabero-VTLA
  commit `1ed9cf44c05bc63fa3b3dbc0ca83ce9dbc8b7b2e`
- Teacher：`NathanWu7/pi0_lora_tacfield_tabero`，step49999，HF revision
  `1ce3438ce5f290da65aff286935ecf97e9ea4046`。
- 场景与初始状态：dataset `NathanWu7/Isaaclab_Libero`，revision
  `fe6263d3ff1f600f123522e298d6d755bc827d33`。
- 机器人/触觉标定：dataset
  `china-sae-robotics/Tactile_Manipulation_Dataset`。
- 上游复现说明：
  https://github.com/NathanWu7/Tabero/blob/f57f147c4abd0c223186878ff595e4a0089c0fbe/docs/REPRODUCTION.md

## 环境版本与已知差异

优先创建独立环境，复现Isaac Sim5.0 + Isaac Lab2.2；不改当前ManiFeel或UniVTAC
环境。上游论文参数是contact_gripper绑定`panda_.*finger`，控制器
`squeeze_ff_k_load_z=0.6`。

当前下载的Tabero源码默认采用minicase复现配置：多处contact_gripper绑定
`gelsight_mini_case_.*`，控制器k=0.9。仅安装论文版本并不会自动恢复论文参数。
请检查实际场景rigid-body prim及配置，记录采用哪个profile。若使用Sim5.1+
Lab2.3，上游记录gelpad绑定问题，minicase是可运行的替代复现协议，但不能把
它的success/force直接报成论文Table3协议。先验证能跑，再固定正式profile。

记录OS、driver、GPU UUID/显存、Python/torch/CUDA和Sim/Lab版本。先检查空闲
GPU与相关进程，再固定模型卡和仿真卡；单环境、headless、enable_cameras。
检查实际Kit renderer GPU和physics GPU均指向仿真卡，模型JAX进程位于模型卡。
CUDA_VISIBLE_DEVICES只控制CUDA可见性，不能代替Kit/Vulkan GPU验证。

## 资产和安装

在独立Tabero checkout安装：

```bash
python -m pip install -e source/tac_manip
python -m pip install -e benchmarks/openpi/openpi-client
```

使用独立数据根，不把权重或大数据写进Git仓库。场景及初始状态仅需：

```bash
hf download NathanWu7/Isaaclab_Libero --repo-type dataset \
  --revision fe6263d3ff1f600f123522e298d6d755bc827d33 \
  --include 'USD/**' 'assembled_hdf5/**' --local-dir /path/to/Isaaclab_Libero
hf download china-sae-robotics/Tactile_Manipulation_Dataset --repo-type dataset \
  --local-dir /path/to/tactile_runtime
```

按上游README建立`benchmarks/datasets/libero/USD`、`assembled_hdf5`链接和
`source/tac_manip/tac_manip/assets/data`链接。注意链接的实际层级要含
`Robots/Franka_gsmini/physx_rigid_gelpads.usd`，以下载树为准。
本3090服务器还需独立安装Tabero-VTLA的JAX server环境，下载Teacher完整params
及原normalizer。只做rollout不需要11GB LeRobot训练数据或原train_state。

```bash
hf download NathanWu7/pi0_lora_tacfield_tabero \
  --revision 1ce3438ce5f290da65aff286935ecf97e9ea4046 \
  --include 'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params/**' \
    'checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/assets/**' \
    'norm_stats/**' \
  --local-dir /path/to/models/pi0_lora_tacfield_tabero
```

策略server与Isaac使用独立环境、独立终端和日志。模型配置为
pi0_lora_tacfield_tabero；不要把通用PyTorch转换结果当作已支持触觉的模型。

## 第一道验收：无策略的传感器smoke

先单任务libero_object task0，单环境100steps，设有界walltime，保存100帧
观测和日志。可参照`force_debug_playground.py`的AppLauncher/gym初始化编写
最小smoke，或用匹配13D replay输入运行该脚本。

重要：assembled_hdf5用于初始状态，不保证含13D hybrid动作。脚本明确检查
dataset action维度等于env action维度，不能把7D/8D assembled动作直接灌入
13D tactile环境。若无13D replay输入，先用正确的当前EEF位姿/夹爪状态构造
neutral13D动作验证渲染，再做一次受控抓取检查接触力。

验收要求：RGB与wrist图像224×224、有限且非全黑；触觉RGB、marker motion随
接触变化；gripper force在确认接触时有响应，离开接触后回落；无NaN/Inf、
相机卡死、contact-sensor绑定失败或GPU错误。仅显示窗口不算通过。
保存RGB/触觉样例、marker数组、左右force时间序列、实际sensor prim路径、
显存峰值和模拟steps/s。首轮shader编译时间与稳定运行时间分开记录。

## 第二道验收：同一3090服务器，分卡部署

在3090服务器的模型终端运行，以下以模型物理卡0为例；改为实际空闲批准卡：

```bash
cd /path/to/Tabero-VTLA
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py --port 18018 policy:checkpoint \
  --policy.config=pi0_lora_tacfield_tabero \
  --policy.dir=/path/to/models/pi0_lora_tacfield_tabero/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999
```

完整params和assets就绪后启动。按上游说明处理checkpoint中的
assets/NathanWu7/tabero_object_25与tabero别名；不要重算normalizer。
等待日志显示server开始监听，并确认一次真实请求能返回有限的H50动作。

在同一服务器的Isaac终端运行，以下以仿真物理卡1为例。CUDA隐藏后的逻辑
cuda:0对应物理卡1；实际Kit/Vulkan设备映射必须单独核实并在必要时通过
--sim-kit-args指定，不能让仿真误占模型卡0。先用--help核对CLI：

```bash
cd /path/to/Tabero
source scripts/tools/set_replay_env.sh inference
CUDA_VISIBLE_DEVICES=1 python scripts/tools/run_task_evaluations.py \
  --policy-model openpi --control-mode tactile \
  --server-host 127.0.0.1 --server-port 18018 --sim-device cuda:0 \
  --task-suites libero_object --task-ids 0 --use-tabero-tasks \
  --num-total-experiments 1 \
  --hdf5-folder benchmarks/datasets/libero/assembled_hdf5 --require-hdf5 \
  --prompt-adverbs firmly --record-step-traces --record-videos \
  --output-dir evaluation_results/tabero_teacher_local_firm --headless
```

随后将prompt-adverbs改为gently，输出目录改为tabero_teacher_local_gentle，
完成第二条。保留完整13D位置/力输出和原混合控制器，不切到abs7d。

此固定commit的effective_max_inference_steps对libero_object硬编码30chunks，
libero_10为50；这是当前官方runner的运行时限，不能仅看CLI默认80，也不能
以传入更大max-inference-steps就声称时限改变。完整episode须按原runner自然
达到成功或timeout，保存最终记录、视频和force traces，不能以手动提前停止
代替。episode运行完成与任务成功分别报告。

检查client发送的是state7、两相机及原触觉历史；force/marker不得含未来信息。
正式subset为libero_object的0,1,2,3,5,6,7,8,9，task4不在subset。

## 单卡备选与显存验收

如仅一张空闲3090，把模型和Isaac都指定到该卡，仍使用两个独立进程和localhost。
保持XLA_PYTHON_CLIENT_PREALLOCATE=false，避免JAX启动即预占大部分显存。
如有OOM，记录分阶段峰值后再调整，不能靠缩小模型、关触觉、禁用相机或更换
权重来声称原baseline可运行。

持续采样各GPU的memory.used和进程表，保存GPU UUID/PID映射。分别记录模型
加载、首请求JAX编译、Isaac初始化/首次渲染、稳定推理和完整episode时的峰值。
双卡方案报告每卡峰值及可用余量；单卡方案报告两个进程共存时的整卡总峰值，
包括JAX/Kit占用，不以PyTorch allocator统计代替。记录server延迟、模拟steps/s、
episode walltime及任何OOM。至少完成firm/gentle各一条完整episode后才判定通过。

## 当前验收状态与回传产物

用户已报告同机双3090部署通过：传感器340steps；firm146steps成功、平均夹持力
48.65N；gentle140steps成功、6.19N；模型GPU3峰值8493MiB、仿真GPU4峰值5542MiB，
通信均为localhost。记录在preparation目录的`rollout_validation_user_report.json`。
这些是用户提供的结果，H200端未独立复跑；无需继续等待H200模拟器验证。

用户指定同步远端https://github.com/ggkiller-air/Tabero。2026-11-03已核对
origin/main为`27d8d7e70ddf16b1e50c5db80adf74912a304a53`
（Fix headless LIBERO tactile evaluation on RTX GPUs），本地与远端一致。
3090 rollout应拉取该fork的main；上方f57f147为原始上游来源记录。

请返回环境和源码版本、GPU型号/UUID/driver、采用paper还是minicase profile、
所有局部patch、单卡还是双卡部署、模型/仿真PID与GPU UUID对应、传感器smoke
报告、firm/gentle完整rollout日志/视频/force traces、是否成功及steps/s/显存峰值。若失败，返回首个关键traceback或GPU/renderer
错误，不要只返回最后一行。已有功能验证结果保留，待同步修复后再对齐新DP/UVIP部署。

NVIDIA关于RT GPU的官方要求：
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html
