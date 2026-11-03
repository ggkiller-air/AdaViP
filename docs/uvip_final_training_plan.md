# UVIP 最终训练方案审查（2026-11-02）

> 2026-11-02后续决策：第三benchmark转向Tabero候选。公开源码、Teacher和数据已找到并启动下载，已完成真实parquet检查；RTX rollout验收待3090端。最新方案见[Tabero可行性](tabero_uvip_feasibility.md)与[RTX交接](tabero_rtx_handoff.md)。下文OpenViTac部分保留为此前审查记录，不继续自动安排其Teacher训练。

> **最终Teacher纠正：ManiFeel使用L1 no-dropout E50 EMA（global_step55700）。**
> 标签为`/data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/original/`，完整验收后才能用于最终Stage2。旧L0 E500与另行生成的L1 E500均保留为历史资产，不作为最终输入。

## 已确认的实验定义

用户确认：DP 使用 L1 no-dropout Teacher；FTP-1 使用自身 Teacher 生成 utility。
三个 benchmark 指 ManiFeel、UniVTAC、OpenViTac。ManiFeel和OpenViTac保留DP+UVIP；
UniVTAC全部实验统一为FTP公开权重覆盖的六任务：insert_hole、insert_tube、
lift_bottle、lift_can、pull_out_key、put_bottle_in_shelf（公开权重名put_bottle）。
不再做insert_HDMI、grasp_classify。UniVTAC FTP明确做六个独立单任务+UVIP，
直接复用公开Teacher，不重新训练Teacher，不合并为联合模型。OpenViTac高级
backbone待取得匹配公开任务权重后确定；原FTP计划不再自动回退到重训Teacher。

统一流程：任务适配的完整输入 Teacher → 冻结 Teacher 生成 signed utility →
Stage 2 utility predictor、模态 FiLM 与 utility-conditioned attention 学习。
本次只审查和制定实施计划，没有提交新训练，也没有改动已有训练实现。

L1 是既有 late-language 模型名称，不指 L1 loss。DP Teacher 保留 action-only
目标、完整输入、无输入 dropout，不加入 Future 辅助目标。FTP-1 保留原生
视觉语言、触觉和 action expert，不强行改名为 L1。

对输入组 m，utility 为同一动作、noise、time 下的损失差：

    u_m(x) = E[loss(Teacher(x without m), a) - loss(Teacher(x), a)]

正值表示该输入降低 Teacher 损失；负值保留。该值衡量模型的预测依赖，不是
环境成功率或物理因果效应。标签保存原始有符号值和估计方差；Stage 2 使用
训练集逐维 RMS 缩放的 SmoothL1，禁止截零、softmax 或重新拟合 Teacher normalizer。

ManiFeel：7 个输入组、10 个条件、9 维 utility、5 个图像 FiLM。
UniVTAC/OpenViTac：6 个输入组、9 个条件、8 维 utility、4 个图像 FiLM。
两个相机 benchmark 的 utility 顺序为 head/wrist/left tactile/right tactile/
state/language/vision/tactile，名称表示移除对应输入。

## S0 三段式原则与独立超参数

| 阶段 | 原ManiFeel S0参考范围 | 训练内容 |
| --- | --- | --- |
| A | epoch1 | 仅 Utility Predictor；冻结 encoder 与 action 路径 |
| B | epoch2–6 | 冻结继承的 encoder；训练 utility、FiLM、query、attention 与 action 路径 |
| C | epoch7–600 | 解冻 encoder，继续联合训练；encoder 使用独立解冻 ramp |

表中epoch是旧S0参考值，不是所有benchmark或backbone的最终预算。阶段长度、
batch、LR、EMA、Utility loss权重和解冻范围按具体模型与数据制定并冻结。

A 使用 Utility loss；B/C 使用 Action loss + 0.05 Utility loss。
Utility 输入 detach，Action 使用的预测 utility detach；Action 仍可以经过
context/FiLM 路径更新 C 阶段 encoder。Stage 2 使用完整输入。

ManiFeel DP的历史参考为batch32、seed42、EMA0.9999、global clip1、500-update
warmup、cosine floor0.1、encoder LR8e-5、utility/action LR1e-4；这些不是其他
benchmark的通用配置。DP action U-Net
重新初始化，继承 Teacher encoder/normalizer。FTP-1 继承完整预训练 action
expert；仅迁移三阶段原则，其参数组、LR 和总更新预算需要按 FTP 规模确定，
不能把 DP 的600epochs/学习率当作已经验证的 FTP 配置。

## 现状和缺口

### ManiFeel DP：最终锁定L1 E50，标签生成中

最终Teacher：
`/data/wangzihao/outputs/manifeel/language_conditioning_t2_batch64_ep500_seed42_20261028_204459/l1/checkpoints/epoch50.ckpt`
身份为E50/global_step55700/EMA，full_input_probability=1.0。

最终标签：
`/data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/original/`
作业416813正在生成，目标65,265窗口。等待该目录COMPLETE和Teacher身份验证，
不使用同run中的dropout10目录，也不使用任何E500 utility数组。

历史资产有两份：旧L0 E500/547400，和另行生成的L1 E500/557000；两份均已
完成。L1 E500曾复核汇总文件哈希，但该结果不构成最终选择E500的授权。
E50可复用既定窗口索引和noise/time采样协议，不能复用E500的utility数值。

现有 `scripts/manifeel/train_stage2_t2.py:load_teacher` 硬编码旧 L0 的
E500/547400；`Stage2Policy` 依赖七 token 的 `encode_tokens/fuse_tokens`。
L1 只有六个非语言 sensor tokens 和 late-language head，直接改 checkpoint
路径会失败。

建议实现独立 L1 adapter：严格恢复 L1 六 sensor encoder、fusion、language
head 和 normalizer；增加512→256的 language token projection供 attention。
明确区分原 L1 的 pre-language z 与最终 condition c，建议 Utility/context
取完整输入 c，attention 使用六 sensor tokens + language token。新增 projection
属于新参数；c 的完整输入输出必须与原 L1 精确一致。这个接法是实施建议，
尚未编码或验证，不能声称旧 S0 已支持它。

DP K64沿用既定分层分配16/12/12/12/12、期望权重.12/.13/.25/.25/.25、两组
独立K32、成对noise/time和严格FP32协议；当前E50标签继续由现有作业完成。

### UniVTAC DP：既有框架可复用，旧 Teacher 不满足新定义

历史数据为8任务、400episodes、75,543 anchors、69,543个T2/H16完整窗口。
最终只取上述六任务，需重新统计anchors/完整窗口、采样权重及训练RMS；不把
历史八任务总量标成最终六任务数据量。
已有 E400 Teacher、utility 和 Transformer Stage 2 基于 full-sensor early-language
且训练有20% masking，不是最终 L1 no-dropout，不能仅重命名后使用。

先迁移四图像+8D state 的 L1 late-language encoder，完整输入训练新 Teacher；
然后生成新 K64 labels，接上述 L1/S0 adapter。保留现有数据颜色、动作对齐、
normalizer和采样约定。旧标签仅供旧实验使用。

### OpenViTac DP：数据已准备，正式流程尚缺几部分

11任务、1,300episodes、306,218帧，T2/H16完整窗口282,967，其中训练255,040、
验证27,927；现有episode split继续保留。H200 smoke报告已存在，完整宽度模型
forward/backward、四HyperNet Stage2检查通过，但optimizer_steps=0，不能据此
声称已完成训练或断点续训验证。

当前 encoder 是 early-language + learned missing tokens；Stage2 是 MLP final
fusion，loss合成为 action+utility，尚非 S0 attention/0.05 loss协议。当前
标签helper仅两次IID draw，尚无正式K64批量生成/分片恢复流程。

需要：L1 no-dropout Teacher、正式K64 archive、S0 attention融合、正确loss权重、
阶段optimizer/scheduler/EMA及完整checkpoint恢复。图像缓存先保持uint8，不能
把接近974GiB的float32数据塞入384GiB主机。

公开数据没有原始action命令，当前使用next observed joints和下一帧left finger
metres作为目标。正式训练前仍需3090控制端核实60Hz、gripper单位与next-state
codec；离线梯度检查无法证明接触任务执行正确。

## FTP-1 checkpoint 审查

公开基础权重：
https://huggingface.co/MJJJJ1064/ftp1_v0426_50kstep

公开 UniVTAC fine-tune 权重：
https://huggingface.co/MJJJJ1064/ftp1_univtac_finetune/tree/main

公开 fine-tune 目录包含 insert_hole、insert_tube、lift_bottle、lift_can、
pull_out_key、put_bottle 六个任务的19999 checkpoint，缺 insert_HDMI、
grasp_classify。它们是六个单任务模型，不是现成的八任务联合 Teacher。

本次下载并读取 insert_hole 的小型配置，未下载或加载大模型：action_dim120、
horizon32、state_input_mode=adarms、bf16、disable_history=true；train配置为
action_joint_rep=mix、non_tactile_dropout_ratio=0.0、20ksteps、batch64、val_ratio.01。
这些配置只对抽查的insert_hole有直接证据，其余五份仍需逐份校验。

因此六任务具备直接打分的候选权重，前提是版本、演示集合、camera、normalizer、
action映射和控制协议匹配。Isaac4.5/5.1、50/100episodes和作者训练split都需核对。
用户已明确采用六个单任务模型，直接逐任务打分和训练UVIP学生，不新增FTP
Teacher训练。逐任务保存normalizer、utility RMS、optimizer与checkpoint。

OpenViTac 仓库 `policy/ftp-1/` 只有客户端；README指向远端
`/notebook/ftp1-policy/checkpoints/finetune/ftp1/<task>_mixnorm_train/39999`。
这是文档约定路径，本次未连接该远端，也未证实实际文件、训练完成或哈希。
现有公开主仓权重表未列OpenViTac fine-tune包。按不重训Teacher的要求，
后续只盘点现成匹配资产，或选择确有OpenViTac任务权重的其他高级方法；
基础pretrain存在不能作为免任务适配的证据。

## FTP-1 utility 与 Stage 2 接口方案

源码：
https://github.com/michaelyuancb/ftp1-policy/blob/main/src/openpi/models_pytorch/ftp1_pytorch.py

1. 打分走模型训练forward，不走websocket infer。forward支持noise/time输入，
   但当前只返回batch聚合loss；需增加逐样本、按有效时间和action slot归一的loss。
   保留120D映射，只让实际有效动作维度参与损失，不平均填充的112个空slot。
2. 使用原生flow matching：x_t=t*noise+(1-t)*a，target=noise-a。
   time来自.001+.999*Beta(1.5,1)，不能直接套DP的0..99 timestep。
   K64可在分位数域分层，保持原生time分布的期望，固定draws并记录估计方差。
3. 打分禁用训练随机增强。eval()本身不够：forward内部仍调用
   `_preprocess_observation(..., train=True)`，必须增加明确的score模式。
4. 对两个vision token块、两个tactile function-area块、language与P分别屏蔽。
   单触觉移除应保留合法结构，清token及pad mask，不能传None触发缺触觉异常。
   P移除必须同时处理state encoder输出和adaRMS conditioning；当前adaRMS使用
   raw state embedding，单独清state mask仍泄漏。
5. FTP采用当前时刻对齐H32、单帧输入和mix动作。重新枚举符合其history/horizon
   的完整窗口，不能使用DP的H16窗口总数，也不能直接搬DP action/normalizer。
   记录chunk[0]当前对齐项的训练mask及execute_from_index，部署保持一致。
6. Stage2在跨模态混合前，对两个相机token块和两个触觉area分别做identity FiLM。
   从不含action/noise/time的观测token提取context，经detach供Utility Predictor。
   用户已确认只新增utility→FiLM，不新增query调整、cross-attention或动作输入
   残差。保留加噪action经120→1024投影形成的原生动作tokens与state/time adaRMS；
   identity初始化FiLM，同noise/time下检查原Teacher输出一致性。
7. 保留FTP原生VLM、tactile expert和action expert。A只训练utility；B冻结观测
   encoder，训练新增模块和action expert；C解冻选定观测模块。全量C与LoRA-C
   属不同参数协议；显存与吞吐测量后冻结选择，不能将LoRA结果称为全量S0。

FTP原生attention在expert层内逐层融合，token宽度也不同；这是新backbone port，
不是把现有DP Stage2 class包在FTP外即可完成。现有权重能省Teacher训练，不能
省上述score与student integration工作。

## 实施顺序与验收

1. 冻结五组run manifest：数据split/版本、任务覆盖、Teacher哈希、normalizer、
   token/utility顺序、动作codec、loss/time分布、训练预算和评测seeds。
   UniVTAC所有实验只报告相同六任务；FTP每任务独立实验。
2. 优先完成ManiFeel L1/S0 adapter并训练；同时迁移UniVTAC/OpenViTac L1 Teacher。
   FTP先用一个公开UniVTAC任务完成原生模型加载、成对打分和初始输出一致性检查，
   然后按已经确定的六个独立单任务扩展，不训练统一FTP Teacher。
3. 标签验收：full条件与native一致；被移除输入任意替换后score不变；batch大小
   改变不改标签；保留负值；K32两半一致性/方差；分片恢复与source join严格验证。
4. Stage2验收：真实A/B/C backward梯度路由、冻结组不变、精确checkpoint续训、
   初始化/normalizer/RMS一致性和EMA；GPU实际吞吐/显存测试后正式长跑。
5. 完成checkpoint脱离Teacher/标签数据的加载与simulation rollout；各方法使用
   相同任务集合和评测seeds。ManiFeel按官方test/mean_score和9任务macro报告。

调度沿用单job的gpu,gpux选择、提交时查询共同最大walltime、per-run锁及断点恢复；
旧job和旧实验资产保持独立。训练时间以实际samples/s及optimizer预算估算，FTP
尚无本地测量；不承诺五组同一天开跑或给出未经测量的FTP完成日期。

审查证据缓存：`/public/home/wangzihao/.cache/uvip-final-audit/`，包含公开文件列表、
FTP小型配置、模型源码和SHA256记录。完整权重加载、FTP吞吐及远端checkpoint
存在性仍待实施验证。


## OpenViTac高级方法公开权重复查（同日用户修订）

本次检查官方README/项目页、GitHub release列表、HF的OpenViTac/OpenVTLA
模型搜索、FVL-Lab模型列表、官方数据集根目录及相关模型作者仓库。HF模型
搜索OpenViTac/OpenVTLA和FVL-Lab模型列表均为空，GitHub releases为空；数据集
根目录为Real_world/Simulation，没有checkpoint目录。这是本次检查结果，不能
证明所有平台永远不存在未链接权重。

| 方法 | 实际发布状态 | 接UVIP判断 | 当前能否免Teacher训练做正式OpenViTac |
| --- | --- | --- | --- |
| OpenVTLA | 官方README仍说model links待发布；有客户端 | AnyTouch2/visual token分组处适合FiLM和utility attention，需完整模型代码 | 未确认匹配任务权重 |
| FTP-1 | 公开pretrain和UniVTAC六任务；OpenViTac只有远端路径说明 | 原生flow loss和expert可接，开发方案见上 | 需现成OpenViTac任务包 |
| N0-VTLA | 有base及NeoSim/UniVTAC任务权重 | flow loss可做utility；DINO差分输入和latent predictor两条路径均需正确屏蔽 | 所查任务包属于NeoSim，不能直接认作OpenViTac |
| InternVLA-A1.5 | 有base/Libero/RoboTwin/DOMINO | 视觉/语言/state可接；原生没有触觉，需额外触觉分支才能检验完整VT UVIP | 未确认OpenViTac任务权重 |

关键排除：NeoteAI/n0_VTLA_grasp_chip、n0_VTLA_pour_ball、n0_VTLA_insert_USB
确实各有model.safetensors和norm资产，但model card明确是NeoSim，评测基于
anlorla/UniVTAC的NeoSim分支commit695a22d。与OpenViTac任务名字相似并不能
证明传感器/场景/动作/初始化与成功条件匹配，不能免验证直接替代。

来源：
- https://github.com/FVL-Repo/OpenViTac#openvtla
- https://github.com/FVL-Repo/OpenViTac/blob/main/docs/Usage.md#baselines
- https://huggingface.co/NeoteAI/n0_VTLA_grasp_chip
- https://huggingface.co/NeoteAI/n0_VTLA_insert_USB
- https://huggingface.co/NeoteAI/n0_VTLA_pour_ball
- https://huggingface.co/InternRobotics/InternVLA-A1.5-base

当前建议：UniVTAC六个FTP单任务优先实施；OpenViTac高级backbone保留为资产
待定，不因基座开源自动新增Teacher重训。若取得作者已有OpenViTac FTP任务包，
优先复用FTP接口；若取得OpenVTLA任务包与完整代码，也适合接UVIP。没有匹配
权重时，公开材料尚不足以给出“下载即打分”的高级方法选项。
