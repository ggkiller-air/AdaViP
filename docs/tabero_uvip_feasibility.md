# Tabero 替换 OpenViTac 的可行性（2026-11-02）

结论：Tabero是可实施的第三benchmark候选，公开资产比当前OpenViTac高级策略
更齐全。建议用DP+UVIP，以及原生Tabero-VTLA+UVIP。没有找到Tabero任务FTP-1
权重，不能把替换后的高级backbone仍标成FTP-1。采用原生Tabero Teacher，无需
为了取得Teacher重新训练。正式替换以RTX传感器smoke和原Teacher rollout为验收。

## 已下载并检查

环境代码commit f57f147c4abd0c223186878ff595e4a0089c0fbe；模型代码commit
1ed9cf44c05bc63fa3b3dbc0ca83ce9dbc8b7b2e。两个官方源码zip已落盘并计算SHA256。

HF公开模型NathanWu7/pi0_lora_tacfield_tabero的49999 params/资产约6.40GB，
revision1ce3438ce5f290da65aff286935ecf97e9ea4046。下载中的完整参数包包含原基座
及LoRA，不只是adapter。已读Orbax元数据，包含LoRA和tactile_prefix_encoder。
不能把“元数据存在”当作完整模型已加载。

训练数据选择当前Field config指定的NathanWu7/tabero_object_25，revision
f468387d8bcabdb53e816650e9ae5c2a7661c2fd，737episodes/109097frames/20Hz。
数据约11.29GB，88个指令条目对应去除gentle/firm副词后的11个基础任务，不是
88个独立操作任务。上游正式评测为9个libero_object任务：0,1,2,3,5,6,7,8,9。
数据集训练覆盖与评测subset要分别记录。

已实际读取一份完成哈希校验的parquet，解码三路RGB224×224并核实有限值：
state7；actions13；force history8×6；marker history9×198×2。state是EEF的
xyz/axis-angle/gripper，不是UniVTAC的关节qpos。actions为7D位置控制项加6D力，
模型输出padding到32D/H50，数据使用绝对→delta等上游transform与原normalizer。

源码、下载清单/哈希、样本报告和持续进度位于：
`/data/wangzihao/outputs/tabero/preparation_20261102/`。
下载状态以download_status.json和DOWNLOAD_COMPLETE.json为准，不宣称尚未
完成的权重或全数据已下载。下载原始train_state不是推理/打分所必需，本次仅
选择完整params与norm/assets；UVIP学生重新建立optimizer。

## UVIP 接口

- 官方JAX Pi0.compute_loss(train=False)返回逐样本逐时间loss；相同rng key
  的noise/time可配对。正式实现可显式传noise/time以稳定batch和分片变化。
- 原生time是Beta(1.5,1)映射，不能复用DP离散timesteps。保存有符号原始差值、
  方差和train-only RMS；不引入Future辅助任务。
- Field模型使用两路相机、一个TCN编码的双指marker-history token、P、L。
  按实际模态定义utility和HyperNet数量，不伪造独立双指encoder。若要双指
  utility，在TCN前同时处理reference/history的对应99个marker，并验证移除
  输入不可泄漏；调制仍应尊重Teacher原来的融合结构。
- Utility/context只取观测特征，禁止依赖加噪动作、time或目标force。保留原
  action expert、LoRA、TCN和13D位置/力输出。A/B/C仅共享冻结顺序，所有预算
  与学习率、force和utility权重独立制定。
- 保留原native action/force/padding loss定义作Teacher full-path基准。当前
  config为effective_action_dim13、force6、weighted_full及默认padding权重；
  不能无说明地换成32维均值或单纯7D位置损失。额外记录position/force分量。
- 源码PyTorch PI0与通用JAX→PyTorch转换脚本没有实现对应tactile TCN路径，
  也不能假定已合并LoRA。首选在原生JAX实现UVIP；若迁移PyTorch，必须完整
  移植并通过权重、完整输入loss与动作的数值一致性检查。
- DP需新Tabero数据adapter和L1 no-dropout Teacher；保留13D输出、force loss
  与同一混合控制器，训练协议不能照搬ManiFeel。不要从原数据猜测joint qpos。

## RTX rollout 是必要验收

按用户最新要求，rollout的模型server与Isaac均部署在同一3090服务器，运行
不依赖H200或用户电脑的网络。优先一张3090跑模型、另一张3090跑单环境仿真，
通过localhost通信；只有一张空闲卡时才测同卡总显存。H200继续负责离线打分
和训练。3090尚未远程实测，不能保证当前driver/环境必然可用。
先单环境100steps验证相机、触觉、接触力，再完成firm/gentle各一条自然结束
的Teacher episode；分别记录各卡峰值、进程映射和episode结果。分卡成功
不能代替同卡24GB显存验收。

上游论文profile为Sim5.0/Lab2.2、panda_.*finger绑定、k=0.6；当前源码多处
默认minicase/k=0.9。Sim5.1/Lab2.3存在上游说明的gelpad绑定问题。必须固定
profile并记录局部patch，不混合两套协议的成功率与力指标。

给3090端Codex的完整操作与验收说明：`docs/tabero_rtx_handoff.md`。

来源：
- https://github.com/NathanWu7/Tabero
- https://github.com/NathanWu7/Tabero-VTLA
- https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero
- https://huggingface.co/datasets/NathanWu7/tabero_object_25
- https://github.com/NathanWu7/Tabero/blob/main/docs/REPRODUCTION.md
