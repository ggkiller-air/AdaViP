# ManiFeel 九任务评测结果与改进计划

## 本次评测的指标判定

**正式报告使用 `test/mean_score` 作为 ManiFeel 基准成功率。**官方评测器先取每个环境在整个 rollout 中的最大奖励，再对环境求平均；环境封装将奖励设为模拟器的 `reset` 标志。九个固定版本的任务都在 `_check_success()` 判定成功或达到 `max_episode_length=100000` 步超时时设置 `reset_buf`。本次评测最多运行 500 步，不可能触发该任务超时。因此，**对这九次评测而言**，`test/mean_score` 恰好是评测期间至少一次达到模拟器任务成功条件的环境占比；它不是失效的代理指标，也不需要改用 `info["successes"]` 重新标注。

**不要用本地 unit runner 额外输出的 `test/success_rate` 替代它。**这个字段根据 `info["success"]` 计算，但普通封装只保留末尾观测窗口的信息，可能漏掉之前发生的成功；ManiFeel 官方评测器不报告此字段。本地另有保留完整 `info` 历史的评测器，其 `test/success_rate` 可以用于诊断，但**不是论文报告字段**。此前 10 环境的 info 派生结果用于诊断和选择 checkpoint，不等于下表正式 50 环境的 `mean_score`。例如，灯泡任务的 80% `mean_score` 与 38% 末尾窗口 `success_rate` 之差，并不说明 `mean_score` 有 bug。

代码依据：`third_party/manifeel/manifeel/env_runner/vistac_pih_runner.py`、`third_party/manifeel/manifeel/envs/vistac_isaacgym_multiple_env_wrapper.py`、`third_party/manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_*.py` 和 `third_party/manifeel/manifeel/config/task/TacSLTask*.yaml`。

## 已选 checkpoint 的正式结果

以下是**九个分别训练的单任务策略**：保存的 EMA 权重、50 环境、最多 500 步、评测种子 100000，统一报告 `test/mean_score`。论文对照取 ManiFeel 表 II 的 **Vision + TacRGB** 行，差值 = 本次得分 − 论文均值，单位为百分点（pp）；只是描述性比较。

| 任务 | 相对于 `checkpoints/` 的路径 | 本次 `mean_score` | 论文 TacRGB（均值±标准差） | 差值（pp） |
| --- | --- | ---: | ---: | ---: |
| 插钉（Peg Insertion） | `0929ours/peg_insertion/latest_epoch100.ckpt` | 34%（17/50） | 21% ± 2% | +13 |
| USB 插入（USB Insertion） | `0929ours/usb_insertion/latest_epoch300.ckpt` | 56%（28/50） | 49% ± 2% | +7 |
| 电源插头插入（Power Plug，HyperResNet Fusion） | `0926/dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt` | **66%（33/50）** | 51% ± 1% | **+15** |
| 齿轮装配（Gear Assembly） | `0929ours/gear_assembly/latest_epoch100.ckpt` | 60%（30/50） | 57% ± 2% | +3 |
| 螺母螺栓装配（Nut-Bolt Assembly） | `0929ours/nut_bolt_assembly/latest_epoch200.ckpt` | 88%（44/50） | 78% ± 3% | +10 |
| 灯泡安装（Bulb Installation） | `0929ours/bulb_installation/latest_epoch200.ckpt` | 80%（40/50） | 72% ± 4% | +8 |
| 插钉重新定向（Peg Reorientation） | `0929ours/peg_reorientation/latest_epoch300.ckpt` | 42%（21/50） | 39% ± 2% | +3 |
| 物体搜索（Object Search） | `0929ours/object_search/latest_epoch300.ckpt` | 100%（50/50） | 69% ± 1% | +31 |
| 球分类（Ball Sorting） | `0929ours/ball_sorting/latest_epoch200.ckpt` | 58%（29/50） | 72% ± 2% | −14 |

原始依据：`/data/wangzihao/outputs/manifeel/eval/original_batch8_best_epochs_n50_steps500_seed100000_20260929/results.csv` 及各行 `eval_log.json`。电源插头在该 CSV 中标为 **Power Plug (fusion only)**，对应 `formal/power_plug_fusion_only_epoch100/eval_log.json`，`test/mean_score=0.66`。它是 **HyperResNet 的融合版本**：`use_fusion=true`、`use_fusion_hypernet=false`，与其余八任务的融合设置一致；“fusion only”并不表示没有对视觉/触觉编码器做 HyperNet 调制。旧 `0929ours/fusionhyper/latest_epoch300.ckpt` 额外启用了融合 HyperNet，得分 56%（28/50）；`batch8_tactile/Film_Fusion/latest_epoch200.ckpt` 是另一个 CLIP/Sparsh FiLM 模型，得分 8%（4/50）。**这三者不可混作同一 checkpoint。**本次选用 epoch 100：旧的 10 环境诊断结果为 9/10，epoch 200/300 均为 8/10；已有 50 环境评测中 epoch 100/200/300 分别是 66%/66%/62%。这些 epoch 共用评测种子，选择后的结果不等于独立测试。

## checkpoint 目录索引

当前 `checkpoints/` 下有 48 个 `.ckpt`，按现有目录区分如下；**不移动或删除大文件**，以免破坏现有评测脚本及保存的配置。上表路径是本次九任务所用的准确文件。

| 目录 | 内容 | 本次用途 |
| --- | --- | --- |
| `checkpoints/batch8_tactile/Film_Fusion/` | 电源插头 CLIP/Sparsh FiLM：epoch 100、200、300 | 另一路线；epoch 200 的 50 环境得分 8%，**不在主表** |
| `checkpoints/batch8_tactile/baseline/` | 电源插头 TacRGB 基线：epoch 100、200、300 | 本地对照，不是 FiLM checkpoint |
| `checkpoints/0929ours/` | 九个任务子目录，各有 epoch 100、200、300 | 使用其中八个任务；`fusionhyper/` 是另一电源插头模型 |
| `checkpoints/0926/` | 五个实验目录，各含 `checkpoints/latest_epoch{100,200,300}.ckpt` | **使用 `dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt`**；其余保留作消融 |

这些大文件受 `.gitignore` 排除，GitHub 同步的是文档和代码，不是 checkpoint 本体。

## 与论文比较的边界

可写作**使用与论文相同的成功率计分定义**，不能宣称实验设置完全一致：论文表 II 汇总 3 个训练种子、50 个初始环境和最后 10 个 epoch；上表每项只有一个 checkpoint。九个模型均使用腕部视觉、TacRGB 和 50 条示范；论文表 VI 对三个探索任务使用前置相机、对螺母螺栓/灯泡使用 30/20 条示范，动作维度也与我们的八任务配置不同。物体搜索 100% 对论文 69% 的差异尤其受到相机视角影响；所有差值仅供定位问题，不能直接归因为方法改进。参见 ManiFeel（arXiv:2505.18472）表 II、表 VI 和图 13。

**物体搜索专项核查（2026-10-10，尚未重新运行 GPU）：**原始 `eval_log.json` 确实记录 `test/mean_score=1.0`（50/50），而末尾窗口的 `test/success_rate=0.82` 不代表只有 41 个环境曾成功。原始录像仅覆盖 2 个环境，不能证明其余 48 个环境的行为。固定任务配置 `TacSLTaskObjectSearch.yaml` 的物体/目标位置和平移初始噪声为零（物体在手中的旋转噪声非零）；模拟器 `_check_success()` 对此任务只检查插头与目标的平均关键点距离落在 **0.12–0.20 m** 之间，不检查抓取、物体类别或稳定保持。这是当前模拟器判定的 100%，**尚不能称为独立核实的物理任务完成率**，也不应把对论文表 II 前置相机 69% 的 +31 pp 写成公平方法增益；论文图 13 对腕部相机的视觉+TacRGB 另报告 92.7%。待指定空闲 GPU 并获得确认后，应使用不同评测种子、50 环境和逐步成功轨迹复测，再做零动作对照；不要改动 pinned `third_party/` 的判定以「修正」既有基准。

## 诊断现状与后续步骤

- **现状：**九个独立单任务 HyperResNet 融合模型；电源插头采用 `0926` 的 epoch 100，50 环境得分 **66%**。旧 FiLM Fusion 的 8% 属于另一模型。checkpoint 选择与正式测试使用同一评测种子，论文对照仅是描述性结果。
- **评测服务器：**优先分析电源插头和球分类的失败轨迹；对成功/失败轨迹标注接近、接触、精细操作阶段，记录门控/模态注意力、实际动作及首次成功时间。热图只反映模态权重，需配合动作敏感性实验；新 GPU 评测须先检查占用并取得设备许可。
- **engram 训练：**在同数据、同种子和 batch 8 下对比融合与无融合、额外融合 HyperNet 和纯视觉/原始 TacRGB 基线，再逐项尝试接触感知门控；本机不开展大规模训练。
- **报告标准：**仍以 50 环境的 `mean_score` 为准，改进模型用独立的 checkpoint 选择/测试种子和多个训练种子复验；分别报告各任务变化，不能仅以平均值或一个任务的提升下结论。
