# archive/ — 早期探索脚本归档

本目录存放已退出生产主线的三个早期探索脚本。它们与生产入口 `run_pattern_nmf.py` **完全相互独立**:互不 import、无共享状态、输出目录互不相交,归档它们不影响 NMF 管线的任何计算。归档于 2026-07-12。

每个脚本头部已加入 `sys.path` 修正,因此仍可从仓库根目录直接运行,例如
`python archive/run_pattern_tucker.py`。

## 脚本清单

### run_pattern_tucker.py
双城市(Baton Rouge / Fort Myers)OD-流量三维张量的非负 Tucker 分解与跨城成分匹配。
这是项目最早的分解路线,后被单城市 NMF(`run_pattern_nmf.py`)取代。
产物写入 `outputs/archive/tucker/` 与 `outputs/archive/tucker_results.pkl`。
独占依赖:`utils/pattern_analysis/matching.py` 整模块、`osm_poi_features.py`(原
`spatial_features.py`,默认开关 `RUN_SPATIAL_FEATURES=False` 关闭)、
`decomposition.py` 的三个 Tucker 函数(`non_negative_tucker_hals`、
`non_negative_tucker_decomposition`、`check_reconstruction_error_detailed`)、
`graph_io.py` 的 `graphs_to_3d_tensor`/`calculate_segment_average`、
`visualization.py` 的四个 Tucker 专属热图/地图函数。

### run_pattern_temporal_decay.py
Baton Rouge 单城市总流量的 Prophet 基线分解与灾后恢复曲线(StepWise)拟合。
产物写入 `outputs/archive/temporal_decay/` 与 `outputs/archive/temporal_decay_results_br.pkl`。
独占依赖:`utils/pattern_analysis/temporal.py` 整模块、`graph_io.py` 的
`calculate_total_flows`、config 的 `PROPHET_FREQ`/`PROPHET_START_DATE`。
**软依赖提示**:`run_prediction_training.py` 的 `temporal_physics` 模型读取本脚本的
产物 `outputs/archive/temporal_decay_results_br.pkl`;当前 outputs/ 中没有该 pkl,因此用
`temporal_physics` 训练前需要先运行本脚本一次(`basic` 模型不需要)。

### run_pattern_distance_decay_paired.py
双城市 normal/disaster 两段各自独立 NMF 后按 cosine 匹配成分,比较配对成分的
距离衰减指数 α 的变化。产物写入 `outputs/archive/distance_decay_paired/`。
独占依赖:`nmf_pipeline.py` 的 `build_paired_matrices`/`decompose_city_paired`、
`visualization.py` 的 `vis_grid_distance_decay`/`vis_slope_paired_alpha`。

## 与生产代码的关系

三个脚本仍从 `config.py` 读取双城市时代的 `BR_/FM_` 常量(生产 NMF 已改用
`run_pattern_nmf.CITY_EVENTS` 五城市-事件注册表,见 `config.py` 头部说明),并
共享 `utils/pattern_analysis` / `utils/data_processing` 中的通用函数。上面列出的
"独占依赖"函数在 utils 中均已标注"仅被 archive/ 下脚本使用",生产读者可以跳过。
运行 tucker 脚本需要 `tensorly` 等生产管线不需要的第三方包。
