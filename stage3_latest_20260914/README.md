# 最新 Stage3：关键网络与核心代码

2026-09-14，按用户要求整理。只复制关键源码、网络配置和必要来源说明；**没有权重、数据集、动作结果、视频、运行环境**。原文件均未修改。

主体对应 2026-09-13 的 CLOSD / UniHSI 物理闭环研究分支；保留其相关的 09-12 目标 DiP / SDF 核心代码作为单独参考，不把几个分支说成已融合成一个新网络。

共复制 **73 个文件，1,118,112 字节（约 1.12 MB）**；每个副本与原文件的 SHA256 已核对。统计不含本目录新增的整理脚本、清单和本说明。

## 目录

| 目录 | 内容 |
|---|---|
| `networks/closd/` | 原 DiP 扩散网络、目标编码、PHC/AMP 控制网络、实际状态反馈任务代码 |
| `networks/unihsi/` | 本地 UniHSI actor / discriminator 网络及场景观测、任务推进核心 |
| `networks/rl_games_core/` | UniHSI 使用的基础网络构建与输入归一化代码 |
| `stage3/` | 我们的 Stage1/2 接口、物理执行入口、SDF 与状态一致性门 |
| `sdf_motion/` | 目标 DiP、keypose / SDF 去噪引导、人体恢复的运动学参考实现 |
| `shared/` | 来源校验、SDF / 旋转表示、人体接口及上游适配等共享核心 |
| `configs/` | 对应网络 / 任务 / 仿真配置；仅参数文本，不含模型权重 |

## 先看这几个文件

1. [DiP 网络与目标编码](networks/closd/closd/diffusion_planner/model/mdm.py)：`MDM`、`TimestepEmbedder`、`EmbedTargetLoc*`。
2. [扩散采样](networks/closd/closd/diffusion_planner/diffusion/gaussian_diffusion.py)：整段动作去噪与采样步骤。
3. [CLOSD 控制网络](networks/closd/closd/learning/amp_network_builder.py)及[真实闭环](networks/closd/closd/env/tasks/closd.py)：DiP 参考 → PHC → PhysX → 已执行状态反馈。
4. [UniHSI 网络](networks/unihsi/unihsi/learning/amp_network_builder.py)：本次实际使用 `AMPBuilder` 的 MLP actor；文件中的其他结构不代表本次均被使用。
5. [最新 Stage1/2 → UniHSI 接入](stage3/unihsi_probe/run_stage12_transfer_v3.py)：路线、座面条件、原 actor 与状态门控 SDF。
6. [状态门](stage3/pd_sdf_filter_v2.py)与[SDF 动作修正](stage3/pd_sdf_filter.py)：先检查 root / rigid-body 状态一致，再小幅修正 PD 控制量。
7. [CLOSD + SDF 执行](stage3/closd_probe/run_closd_convex_sdf.py)与[未来参考约束](stage3/closd_probe/convex_reference_sdf.py)：在去噪内约束尚未执行的参考动作。
8. [目标 DiP / keypose 参考入口](sdf_motion/target/run_target_keypose_pilot.py)：此前不使用物理 actor 的运动学分支，和上面的物理闭环分开阅读。

## 使用与来源边界

- 这是**源码阅读 / 保存包，不是脱离原工作区即可运行的完整软件包**。代码保持原字节，原路径、导入、权重及模拟器依赖没有为本目录重写；运行仍应从原方法目录和已有虚拟环境进行。本次没有训练或推理。
- 最新原入口：`../methods/paper_structure_stage3_20260913/`。两个前序目录为 `../methods/closd_unihsi_sdf_stage3_20260912/` 与对应 `_v2/`。
- UniHSI 已验证的是本地 MLP checkpoint 的加载和计算，官方权重来源未核实，论文 CNN 未完整复现；不能因复制了网络定义便称两篇论文完整复现。当前 Stage1/2 部分迁移仍未完成交互，完整 135D keypose / 体型 / 终朝向也未全部接入 actor。
- `networks/` 包含第三方参考源码，不全部属于用户原创代码。保留现有许可证、文件头与来源 README；本次仅本地整理，未上传 GitHub。
- [逐文件原路径与 SHA256](MANIFEST.json)、[复制核验](VERIFICATION.json)、[明确白名单的整理脚本](collect_core.py)。原权重仍在原处，本目录没有复制，也不通过链接引入权重。

之前误开始的完整整理副本已在本目录重建前清除，只删除本次新产生的副本；所有原代码、原权重和原实验结果都保留，可从原处重新复制。
