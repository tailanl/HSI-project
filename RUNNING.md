# 运行说明

需要 Python 3.10+。先独立安装模型依赖并准备权重，再安装本项目。`pyproject.toml`
并不尝试自动安装所有第三方 CUDA 栈，也不会下载权重。

```bash
python -m pip install -e '.[geometry,guidance]'
python -m hsi --help
```

## 1. 场景理解与规划

首次场景用 `scene-build`；已完成 Stage1A 时，直接向 `plan --scene` 提供保存的 FINAL 收据。
两者均需要显式 Qwen endpoint 和服务端实际 model ID。Qwen API key 只用 `--api-key-env`
指定环境变量名称，不要把 key 值写进参数、JSON 或源码。

```bash
python -m hsi scene-build --help
python -m hsi plan --help
```

`configs/perception.example.json` 描述外部 render/SAM 环境。配置中的 GPU 与 `scene-build --gpu`
应一致。Stage1B 输出 `receipt.json`，每个交互的输入交接在
`interactions/step_00/stage1_execution.json`，对应描述在该步骤的 `keynode_descriptions/receipt.json`。

## 2. 图片 keypose

`stage2 --stage1` 接收单个交互的 Stage1 收据，而不是顶层 sequence 收据。
`--render` 接收同一场景的 48 视角 render 收据；`--descriptions` 接收该交互的关键点描述收据。

```bash
python -m hsi stage2 --help
```

`configs/recovery.example.json` 给出 HybrIK-X 必需路径；其中 `hybrik_config` 必须位于
`hybrik_root` 内。`--comfy` 与 `--models` 是外置 ComfyUI 和模型根目录。
`--segmentation` 是外部 SMPL-X 顶点分区 JSON。主流程不会在当前进程里常驻多个生成模型，
GPU/EGL 工作在独立进程中执行；没有远程作业队列。

Stage1A 可通过 `hsi.memory.facts.build_scene_facts` 生成并保存描述。
完整视角搜索的选择可通过 `hsi.memory.views.save_selection` 保存；后续
`stage2 --mode memory --facts ... --memory-store ...` 使用提示并重新做几何检查。
NavMesh 图不进入 Memory。

## 3. 动作生成

```bash
python -m hsi compile-motion --help
python -m hsi motion --help
python -m hsi evaluate-motion --help
```

`compile-motion` 只接受通过全部审核的 Stage2 结果；当前支持单次坐姿交互。
外部 ReMoGen 路径见 `configs/motion.example.json`，参数旁的 `args.yaml`、统计量与文本缓存也必须存在。
`motion --gpu` 指定一张卡。动作生成成功后仍须 `evaluate-motion` 的全身网格评估，不能直接记为发布通过。
评估需要明确的 `--mean-hands` 资产及 `configs/metrics.example.json` 对应的外部指标源。

扩展目录中的研究代码保留了尚未完全联调的当前 Memory 入库／几何快速分支。
不要把它们的数值候选、历史通过结论或示例当作主流程已经通过的完整结果。
