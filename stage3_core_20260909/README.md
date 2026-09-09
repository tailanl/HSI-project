# Stage3 关键代码摘录

这是刚备份的 **R4 large＋v2接触策略** 的代码阅读包：22个完整原文件，另按9个主题摘出50个类/函数。不是新改版、伪代码、重新训练结果，也不是把旧ReMoGen当成当前网络。

**先看这三个：** [Keypose与上下文](excerpts/02_keypose_context.md) → [整序列去噪主干](excerpts/04_denoiser.md) → [实际采样循环](excerpts/05_sampling.md)。

## 按模块阅读

| 顺序 | 关键代码 |
| --- | --- |
| 1 | [场景：四路窄维编码与局部几何聚合](excerpts/01_scene.md) |
| 2 | [Keypose、语义变化、部位权重与双向上下文](excerpts/02_keypose_context.md) |
| 3 | [初始时间分配与循环中的有界时间反馈](excerpts/03_timing.md) |
| 4 | [16层整序列扩散去噪主干](excerpts/04_denoiser.md) |
| 5 | [DDIM完整采样循环与末端refine入口](excerpts/05_sampling.md) |
| 6 | [v2接触与脚滑规则：不提前锁住未来落脚点](excerpts/06_contact_policy.md) |
| 7 | [权限投影、全序列refine与失败回退](excerpts/07_constraints_refine.md) |
| 8 | [联合训练损失与真实运动监督](excerpts/08_training.md) |
| 9 | [LINGO动作与无时间戳条件的构造](excerpts/09_data_condition.md) |

每份摘录保留原文，标明完整文件和原行号；没有为了好看删除函数内部的检查或改写算法。仅略去不在所选符号内的导入/其他函数，不能把摘录块直接当独立模块执行。

## 真正的当前调用链

```text
JointSequenceDenoiser（16层、512维）
 ├─ encode → JointContextEncoder.forward
 │           ├─ 四路128维部位/场景编码
 │           ├─ root/关节/接触/文本 + 原始变化/语义变化
 │           └─ 双向fusion → 条件tokens、场景tokens、初始时间表
 └─ denoise → WholeSequenceDenoiser.denoise
              └─ SequenceBlock / ExpandedSequenceBlock
                 时间自注意力 → 场景 → 动态关系 → 带时间偏置的上下文

stage3_joint_v2.sample_motion
 └─ 反复：denoise → 时间反馈 → 权限投影/几何guidance
                   → 当前关系查询 → DDIM更新整段动作
stage3_joint_v2.sample_joint_motion
 └─ sample_motion → 全序列refine → 全网格审核/回退 → 动作与实际keyposes
```

事件时间、静态场景与动态身体关系不是同一个输入；时间自注意力也不是预测的事件时间表。静态场景tokens只编码一次，但身体—场景关系与时间反馈进入采样循环。

## 最关键的原始文件

- [R4模型与上下文](source/hsi/stage3_joint/model.py)：当前空间邻域、部位权重、语义变化、上下文和大模型构造。
- [基础编码器与去噪器](source/hsi/stage3_sequence/model.py)：四路编码器、场景池化、SequenceBlock、实际继承的denoise。
- [动态时间反馈](source/hsi/stage3_sequence/timing_feedback.py)。
- [当前v2采样](source/hsi/stage3_joint_v2/sampling.py)、[当前v2 refine](source/hsi/stage3_joint_v2/refine.py)。
- [当前v2接触规则](source/hsi/stage3_joint_v2/contacts.py)、[当前物理目标](source/hsi/stage3_joint_v2/objectives.py)。
- [当前联合损失](source/hsi/stage3_joint_v2/training.py)、[正式训练入口](source/experiments/stage3_joint_v2_train.py)。
- [实际训练配置](model_config.json)：从备份run.json提取，K≤3、T≤180、D512、16层、4×128、约1.04亿参数；不是基础类默认参数。

## 同名旧函数不要混淆

1. 当前上下文前向是 **JointContextEncoder.forward**，不是父类ContextEncoder.forward。后者完整原文件中仍存在，但不用于本模型的encode。
2. 当前场景local是 **SpatialNeighborhoodAggregation**；旧LocalPointAggregation不是R4实际启用的邻域实现。
3. 当前训练损失来自 **stage3_joint_v2/training.py**。旧stage3_joint/training.py只复用JointLossConfig、GeometryCache、masked_contact_loss、derivative_losses与perturb_allowed_keyposes；旧foot_slip_across_events和旧joint_training_loss不是本次入口。
4. 当前循环、物理接触目标与末端refine来自 **stage3_joint_v2**。基础sampling只提供SamplingConfig，基础objectives仍提供权限投影/keypose工具/缓存，不能将其旧physical_objectives当成v2逻辑。
5. GeometryCache复用旧refine中的mesh_in_chunks工具，不代表启用了旧refine流程。
6. 建立接触的目标意图与当前已建立支撑分开。required未满足仍有损失；foot_slip为0且有效支撑对为0时，不是动作成功。

## 范围、来源与校验

- source/ 是22个完整原文件，逐字复制，保持原目录结构和原始行号；没有修改imports或覆盖原函数。
- excerpts/ 是按符号摘出的阅读材料；[excerpt_manifest.json](excerpt_manifest.json)记录每个符号的来源、行号与SHA256。
- [source_SHA256SUMS](source_SHA256SUMS)校验22个原文件；在source/执行 `sha256sum --check ../source_SHA256SUMS`。
- 全部源来自[原备份](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/backups/stage3_current_20260909_141013/README.md)。没有复制权重、数据、大量测试、渲染产物或第三方资产；需要完整执行请使用[完整版备份code/](/home/lzsh2025/kimodo-viser/kimodo_scene_project/agent9/backups/stage3_current_20260909_141013/code/)或原项目及原环境，**不要把这个阅读包当作独立可运行发行包**。
- 新版训练和采样有上述结构，不代表步态、接触或避障已经达到质量要求。本次只检查摘录正确性/语法，不导入模型、不占GPU、不停止训练。

源文件、原备份、正在训练的代码均不修改。
