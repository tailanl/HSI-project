# HSI Project

当前方法的整合工程。源码按职责组织，不再按实验日期、agent 工作区或历史版本入口组织。

```
hsi/
  stage1/   固定场景理解、Qwen 语义规划、接触区域与冷 NavMesh 路线
  stage2/   当前视角审查、H3 原生图片、HybrIK-X、几何 refine 与审核
  stage3/   顺序 keypose 条件、ReMoGen 采样、SDF 与接触引导、物理评估
  memory/   场景事实、相机提示、局部经验及事务存储
  common/   严格文件绑定与不可变收据
tests/      CPU 契约、数值与集成边界回归
configs/    外部环境与模型路径的配置模板
extensions/ 当前研究扩展源码（尚未与新入口完成联调）
```

## 保持的方法约束

- Stage1A 与任务文本解耦，完成后持久化；物体类别不由路线方向覆盖。
- Stage1B 的 Qwen 只做语义与 ID 选择，几何程序计算坐标；每次重新构建 NavMesh，不保存或复用图。
- 相机 Memory 只提供 hint，新任务仍做当前 mask、crop、owner 与 depth 检查，失败回退完整搜索。
- Stage2 使用 H3 原生 `T=1`、512×512、8 steps 图片，不用视频抽帧；生成图片不等于通过 keypose 审核。
- Stage3 保留完整场景 SDF 和原有物理门；局部接触引导不是全身避障证明。
- 不用历史完整 pose/motion 或旧 Qwen 通过结论授权新任务。生产成功经验需要完整、注册过的真实执行证据；目前整合的存储入口不允许伪造正经验。

## 运行边界

项目不包含数据集、权重、第三方模型实现、个人路径、密钥和运行产物。
Qwen 需要显式服务 URL 与模型 ID；ComfyUI/SAM、HybrIK-X、SMPL-X、ReMoGen 需要显式外部安装及模型路径。
API key 只通过环境变量名指定，不能写进配置或提交仓库。

```bash
python -m pip install -e '.[geometry,guidance,test]'
python -m hsi --help
python -m pytest -q
```

第三方 GPU 环境分离执行，H3 与 HybrIK 不应在已提前加载其他 Torch 模型的进程中启动。
这些命令不自动下载任何模型，也不启动长期服务或提交远程队列。

## 使用入口

`python -m hsi --help` 提供各阶段的统一入口：

- `scene-build`：Stage1A 场景理解；`check-scene`：检查已保存的 FINAL 场景。
- `plan`：基于已理解场景进行 Stage1B 规划，保存分步路线和关键点描述。
- `stage2`：完整的当前图片 keypose 流程。`views / image-job / image / recover` 可单独调用。
- `compile-motion / motion / evaluate-motion`：严格交接、当前动作生成和全身网格评估。

各模块 README 说明实际参数与限制；`configs/*.example.json` 只含占位路径，需替换为自己的外部安装路径。
目前主数值执行器支持坐姿交互；规划能保留同一目标的 walk→sit 语义，不代表任意动作已经实现。

## 代码保留与验证边界

本目录以完整保留当前代码、集中组织为目的，不包含历史实验结果或多套历史工程目录。
主流程采用普通包内导入。尚未完成新入口联调的当前生产 Memory／几何 keypose 研究代码保留在
`extensions/affordance_memory/`，不是可直接替代主入口的已验证组件；具体依赖见其 README。
主包的生产成功经验授权尚未接通时会明确拒绝，不能用一个通过标记伪造成功入库。

在停止扩大测试前，最近一轮 CPU 回归为 **538 项通过、1 项跳过**（测试环境缺少 `pathfinder`）。
这之后的收尾改动没有再次完整回归。已做部分新旧数值对照，但**没有重新执行完整 GPU 端到端测试**，
也不声明新的耗时或精度提升。按当前要求，不继续进行完整测试。

原始工作区和已有备份未修改；GitHub 上传与此本地代码整理分开进行。
