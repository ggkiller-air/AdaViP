# UniVTAC / Tabero UVIP 结构确认稿（2026-11-02）

本文件保留 11 月 2 日的设计讨论。当前执行协议与审查结果以
`logs/2026-11-03-native-uvip-integration.md` 和
`logs/2026-11-04-three-benchmark-review.md` 为准：原生 FTP/VTLA 均仅新增
utility/context/FiLM 并训练 action LoRA，C 阶段保持冻结范围、降低 LR；
Tabero-VTLA 没有这里早期建议的额外 attention/query/action residual。
当前 DP Teacher 为 ManiFeel E50、UniVTAC E300、Tabero E150 的 EMA。

状态：用户已确认FTP保留utility→FiLM，删除新增query/attention/动作残差；
Tabero已确认Field+FS输入方案。确认设计不代表Stage2已开训。
UniVTAC DP L1已经按既定late-language/no-dropout定义实现，独立快照CPU验证通过。
ManiFeel最终仍使用正在生成的L1 no-dropout E50标签，不替换成任何E500。

## 当前可以推进的四条路线

| 路线 | 资产与依赖 | 下一训练阶段 |
| --- | --- | --- |
| UniVTAC DP | 6任务/300episodes/67040anchors；作业417405已通过实际100updates检查 | 正在训练action-only Stage1 Teacher |
| UniVTAC FTP-1 | 6个单任务原生Teacher已发布，完整权重下载中 | 先每任务utility，再原Teacher上finetune UVIP学生 |
| Tabero DP | Field adapter/缓存/真实数据检查/精确续训通过；已提交417432 | action-only L1 Stage1 Teacher |
| Tabero-VTLA | Field+FS49999完整params/norm到齐；用户报告3090 rollout通过 | 先utility，再原Teacher上finetune UVIP学生 |

后两种高级模型不重训Teacher；“finetune”是Stage2学生学习，必须等待其对应
utility完成。DP Teacher训练与这些打分任务可并行；不依赖H200端模拟器。

## UniVTAC DP：已确定的L1 Stage1

- 六任务：insert_hole/insert_tube/lift_bottle/lift_can/pull_out_key/put_bottle_in_shelf。
- V1/V2/T1/T2：每路[B,2,3,256,256]，四个独立ResNet18/GN，逐帧512D，
  两帧拼成1024D后投影到256D。
- P：[B,2,8]（7关节+1夹爪），16→256D。
- 五个非语言tokens：[B,5,256]，sensor identity后拼接5个presence flags，
  1285→256→256，得到不含语言的z。
- L：冻结CLIP512D，late head将[z,L]的768D映射256→256D。
- DP：同ConditionalUnet1D宽度512/1024/2048，H16×8D，执行8步。
- 完整输入训练、无dropout、无Future头、无utility头。训练完选定Teacher后再
  做九条件/八维utility。原八任务Teacher/normalizer不能直接重命名复用。
- 本次独立训练配置：batch64、FP32、AdamW8e-5、warmup500、500epochs，保留
  每50epochs快照与latest。其首个100updates是实际训练的性能检查；通过后
  原地继续。这里是UniVTAC DP的具体配置，不用于约束FTP或Tabero预算。

## UniVTAC FTP-1：已确认的Stage2结构

原生结构由源码和checkpoint配置核实：VLM gemma_2b宽2048、action expert
gemma_300m宽1024、tactile expert gemma_small宽512，三分支均18层。
触觉不是1024D。保留原视觉、T3/HPT tokenizer、function-area embedding、
多expert注意力和action head，不用新DP替换原动作模型。

### 输入与输出

| 项目 | 尺寸/语义 |
| --- | --- |
| 两相机 | 两路单帧224×224 RGB；每路SigLIP patch block宽2048 |
| 左右触觉 | 单帧[B,1,2,224,224,3]；HPT产生[B,48,512]容器，当前两pad位置有效 |
| P | 8个物理关节值映射到120D容器，Fourier state encoder宽1024 |
| L | 原生tokenizer，最多150tokens，每个2048D；不是CLIP512输入 |
| action hidden | [B,32,1024] |
| action output | [B,32,120]；实际7关节+1夹爪维有效，保留原mask和mix规范 |

每份checkpoint的具体camera/domain/normalizer/action slot仍逐份校验，不仅
以抽查insert_hole推断所有任务完全相同。左右指在function-area空间区分，
48个容器位置不等于48个实际触觉传感器。

### 新增UVIP模块

1. 在expert跨模态融合之前，分别pool V1/V2/T1/T2/P/L的观测特征，使用六个
   投影层将2048/2048/512/512/1024/2048D各映射256D，得到[B,6,256]。
   特征不能含noisy action、diffusion/flow time或target action。
2. 六tokens加6个presence flags，1542→256→256得到context c。
3. stopgrad(c)→MLP256→256→8，预测归一化signed utility。
   8维顺序为V1,V2,T1,T2,P,L,vision,tactile。
4. context projection256→64与stopgrad(utility)的8→64拼成128D HyperNet条件。
   四个独立FiLM：两相机输出各4096D(gamma/beta各2048)；两触觉输出各1024D
   (gamma/beta各512)，在对应native token block上施加identity初始化FiLM。
   P/L不经FiLM。
5. 调制后的原生视觉/触觉tokens交给原生多expert attention。P/L保持原生路径。
   不新增utility query、额外cross-attention或256→1024动作输入残差。
6. 原生action输入维持x_t=t*noise+(1-t)*action，经原action_in_proj(120→1024)
   得到H32动作tokens；time与state继续走原adaRMS。action expert内部仍由这些
   动作tokens形成Q，读取调制后的观测。输出保持H32×120。
   新FiLM以identity初始化，相同noise/time下须恢复Teacher输出。

### 打分、训练和梯度

Teacher使用原生flow目标，保持8个有效action维度与完整时间mask，成对共享
noise/time；时间分布为.001+.999*Beta(1.5,1)。禁用随机图像增强。P移除同时
清Fourier输出和adaRMS里的state分量，不能只清state mask。

A只训练utility；B冻结继承的观测分支，训练新增UVIP与action expert；C再解冻
选定观测参数。Utility loss不更新encoder；action梯度不进入utility predictor。
全量/LoRA解冻范围、batch、分组LR、阶段步数、EMA在实际显存/吞吐检查后单独
固定，不预设600epochs。每任务独立学生、标签、RMS、normalizer和checkpoint。

## Tabero：已确认匹配Field+FS输入

用户回报：传感器340steps通过；firm146steps成功/平均夹持力48.65N；gentle
140steps成功/6.19N；同机localhost，模型GPU3峰值8493MiB，仿真GPU4峰值5542MiB。
这是用户提供的验证结果，原日志/补丁尚未同步到当前H200代码目录。用户指定
远端https://github.com/ggkiller-air/Tabero；已克隆到`/public/home/wangzihao/projects/Tabero`。
查询时仅有main，HEAD=f57f147c4abd0c223186878ff595e4a0089c0fbe，与先前源档一致，
还不能将远端这个版本认作已经包含3090修复。待用户提供已推送的分支/commit。

### DP L1 Stage1实现

- V1/V2：两帧相机，独立ResNet18/GN→256D各一token。
- T：当前时刻已有的causal marker history[B,9,198,2]（reference+8历史帧），
  reshape[B,9,396]，一个TCN→256D；保留双指合并的native Field语义。
- P：两帧7D EEF xyz/axis-angle/gripper，14→256D。
- 四个非语言tokens形成1028→256→256的z；L为冻结CLIP512D，late768→256→256。
- 输出H16×13D（7D位置/夹爪+6D力），执行与原相同混合控制器。position和force
  loss分别记录、权重明确；不把13D目标裁成7D，也不改成UniVTAC关节action。
- 全输入、无dropout、action/force-only，无Future或utility辅助头。
- 737episodes/109097frames/88prompts全部保留；两路RGB uint8缓存，marker保留全部
  reference+8帧，不做reference subtraction。只在数据层转换动作前6D为相对当前
  observation anchor的delta；夹爪和6D力保持绝对值。normalizer按真实训练窗口拟合。
- 当前DP预算独立设为batch64/FP32、AdamW1e-4、warmup1000、200epochs/341000updates，
  每25epochs保留快照，供后续RTX选Teacher；并非承诺E200最佳。force权重0.1，
  loss=(7*position_MSE+0.6*force_MSE)/13，分别记录两部分。此13D DP目标与VLA
  的32D原生padding目标各自保留，不能跨backbone直接比较utility绝对大小。
- 首100updates是实际训练与性能门限检查，通过后原地继续；CPU精确续训检查
  覆盖模型、EMA、Adam。checkpoint保存config/normalizer/RNG/数据cursor并严格校验。

### 原生Tabero-VTLA Stage2建议

- 保留π0 LoRA模型、两视觉patch blocks2048D、Field TCN产生的一个2048D token，
  原生language tokens2048D（π0配置max48），以及1024D state/action expert。
- 原生H50×32D输出，前13D有位置/力语义；沿用原normalizer与loss配置。
- 采用五个观测组V1/V2/T/P/L，五组投影到256D；新增utility为6D：
  V1,V2,T,P,L,vision。tactile单组与T移除相同，不重复加一个虚假维度。
- 三个native宽2048的FiLM对应V1/V2/T；utility-conditioned4-head attention
  产生256D context，经零初始化256→1024残差接回原action路径。
- UVIP在原生JAX实现。通用PyTorch转换脚本不覆盖TCN，不能直接转换后声称一致。
- 同时保留signed utility的native loss与position/force分量，A/B/C阶段顺序同上；
  LoRA/encoder解冻范围、更新预算和学习率针对该模型另定。

若用户选择Tabero DP改用触觉RGB，应明确这是另一输入协议；Field Teacher与DP
各自产生自己的utility，不把Field标签直接充当RGB模型的utility。

## 需要对齐的3090产物

请提供通过验证的环境版本、paper/minicase profile、Tabero与Tabero-VTLA提交或
补丁、控制参数、Teacher来源及normalizer配置路径。H200离线学习不等待联网，
但最终部署loader和动作/force语义必须与那边修复版本一致。
