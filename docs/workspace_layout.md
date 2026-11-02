# 工作区组织

日常开发使用三个独立 Git 仓库，改动直接提交到各自的 `main`：

| 本机目录 | GitHub | 职责 |
| --- | --- | --- |
| `/public/home/wangzihao/projects/AdaViP-main` | `ggkiller-air/AdaViP` | AdaViP 方法及 ManiFeel、OpenViTac 适配 |
| `/public/home/wangzihao/projects/UniVTAC` | `ggkiller-air/UniVTAC` | UniVTAC 实验与评测 |
| `/public/home/wangzihao/projects/OpenViTac` | `ggkiller-air/OpenViTac` | 官方 OpenViTac 环境与 `experiments/ACTPullDrawer` 基线 |

三个仓库分别维护环境、数据约定和版本历史。数据、checkpoint、运行日志继续
使用 `/data/wangzihao/` 下的独立目录；本次整理不移动或重启训练任务。

AdaViP-main 已从 linked worktree 转为独立 checkout，保留原有分支、标签和
远程引用。三个已初始化的 submodule 也有独立对象库，不再借用旧实验目录。
原 OpenViTac 适配提交已快进合入现有 `main`，无需切换旧 benchmark 分支。

## 旧目录

2026-11-02 的归档根目录：

```text
/public/home/wangzihao/project-archive/2026-11-02/
├── legacy/
│   ├── AdaViP/                     # 原主 worktree、原 .git 与旧 outputs
│   ├── Piper_adavip/               # 保留代码、CLIP/Sparsh 依赖和权重
│   └── OpenViTac-ACT-pull-drawer/   # 导入 fork 前的完整独立仓库
├── worktrees/
│   ├── AdaViP-openvitac/
│   ├── AdaViP-stage2-mlp-fusion/
│   └── AdaViP-t2-utility-fusion/
├── artifacts/
└── metadata/
```

归档保留全部文件和旧仓库历史，含 T2 工作区原有未提交日志、图片与 JSON。
归档 worktree 的 Git 指针和 submodule 路径已修复；它们仍共享
`legacy/AdaViP/.git`，应整体保留。旧日志记录的是当时的路径，不回写历史。

活跃脚本的仓库路径已改为 AdaViP-main。旧 utility labels 继续引用归档中的
`legacy/AdaViP/outputs/`；CLIP/Sparsh 配置继续引用归档 Piper 依赖，支持原有
环境变量覆盖。归档内的历史启动脚本保留原文，重新运行前需指定当前路径。

归档根目录的 `before.json`、`after-move-verification.json` 记录迁移前后的
HEAD、工作区状态及跟踪文件 SHA256；`updated-entrypoints.json` 记录路径修正。
Piper_adavip 没有本地 Git 元数据，也未找到 GitHub 关联，因此没有创建或推送
新仓库。
