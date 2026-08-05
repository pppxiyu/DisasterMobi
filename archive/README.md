# archive/ — 早期探索脚本归档

本目录原有三个已退出生产主线的早期探索脚本(归档于 2026-07-12)。**2026-08-05
清扫**删除了其中两个(`run_pattern_tucker.py`、`run_pattern_distance_decay_paired.py`)
及它们在 `utils/` 中的全部独占依赖;完整代码保存在 git 标签
`full-code-2026-08-05`,恢复方式:

    git checkout full-code-2026-08-05 -- <path>

保留的脚本与生产入口 `run_pattern_nmf.py` 完全相互独立:互不 import、无共享
状态、输出目录互不相交。脚本头部有 `sys.path` 修正,可从仓库根目录直接运行:
`python archive/run_pattern_temporal_decay.py`。

## 脚本清单

### run_pattern_temporal_decay.py(保留)

Baton Rouge 单城市总流量的 Prophet 基线分解与灾后恢复曲线(StepWise)拟合。
产物写入 `outputs/archive/temporal_decay/` 与
`outputs/archive/temporal_decay_results_br.pkl`。

保留原因:它是 GRU 预测管线 `run_prediction_training.py` 的 `temporal_physics`
模型输入 pkl 的**唯一生成器**,不属于"只生成 outputs/archive 结果的代码"。
当前 outputs/ 中没有该 pkl,用 `temporal_physics` 训练前需要先运行本脚本一次
(`basic` 模型不需要)。

独占依赖(utils 中保留,均已标注):`utils/pattern_analysis/temporal.py` 整模块
(其中 `StepWiseModel` 类同时是 GRU 反序列化该 pkl 时的硬依赖)、`graph_io.py`
的 `calculate_total_flows`、config 的 `PROPHET_FREQ`/`PROPHET_START_DATE`。

### run_pattern_tucker.py(已删除,见标签)

双城市 OD-流量三维张量的非负 Tucker 分解与跨城成分匹配——项目最早的分解路线,
后被单城市 NMF 取代。随之删除的独占依赖:`matching.py`、`osm_poi_features.py`
两个整模块、`decomposition.py` 的三个 Tucker 函数及 tensorly 依赖、`graph_io.py`
的 `graphs_to_3d_tensor`/`calculate_segment_average`、`visualization.py` 的四个
Tucker 专属图函数。

### run_pattern_distance_decay_paired.py(已删除,见标签)

双城市 normal/disaster 两段独立 NMF、cosine 匹配后比较配对成分距离衰减指数的
变化。随之删除的独占依赖:`nmf_pipeline.py` 的
`build_paired_matrices`/`decompose_city_paired`、`visualization.py` 的
`vis_grid_distance_decay`/`vis_slope_paired_alpha`。
