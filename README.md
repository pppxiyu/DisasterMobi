# Research_DisasterMobi

Cuebiq OD 流量数据的灾害移动性研究:对五个飓风城市-事件的 OD 流量矩阵做 NMF 分解,
提取逐成分的时间/功能/空间/社会经济/韧性特征,并做跨城市的城市级韧性(`cum_loss`)预测。

## 仓库地图

### 生产入口(在役)

| 文件 | 作用 |
|---|---|
| `run_pattern_nmf.py` | **生产主管线**:五城市-事件 NMF 分解 → 逐成分特征 → 城市内韧性回归 → 跨城市留一预测与城市级重建。城市-事件注册表 `CITY_EVENTS` 在本文件内定义。产物写入 `outputs/nmf/`。 |
| `tune_nmf_optuna.py` | NMF 超参数 Optuna 调参器;`import run_pattern_nmf as rp` 复用其常量与函数(单向依赖)。 |
| `run_prediction_training.py` / `run_prediction_eval.py` | GraphGRU 流量预测管线(Baton Rouge 单城市);`temporal_physics` 模型读取 `outputs/temporal_decay_results_br.pkl`(由归档的 temporal_decay 脚本产出,pkl 已存在)。 |
| `config.py` | 共享路径与时间槽常量。注意双机制:生产 NMF 用 `CITY_EVENTS`,`BR_/FM_` 双城市常量仅服务归档脚本与预测管线(见 config.py 头部说明)。 |

### archive/(已退役的早期探索脚本)

`run_pattern_tucker.py`(双城市 Tucker 分解)、`run_pattern_temporal_decay.py`
(Prophet 基线 + 恢复曲线)、`run_pattern_distance_decay_paired.py`(配对 NMF 距离衰减)。
与生产管线完全相互独立,仍可从仓库根运行;详见 `archive/README.md`。

### 工具包

| 包 | 内容 |
|---|---|
| `utils/pattern_analysis/` | 分解算法(`decomposition.py`)、管线编排(`nmf_pipeline.py`)、图加载与矩阵转换(`graph_io.py`)、逐成分特征(`component_features.py`)、O×D 功能交叉表(`space_function.py`)、韧性回归与跨城市迁移(`ml_resilience.py`)、绘图(`visualization.py`)。仅服务归档脚本的模块:`matching.py`、`temporal.py`、`osm_poi_features.py`(各自头部有标注)。 |
| `utils/data_processing/` | 几何加载(`geo_loader.py`)、EPA SLD 土地利用抓取与 TF-IWF 分类(`fetch_sld_landuse.py`)、ACS 收入抓取(`fetch_acs_income.py`,需 Census API key,见其头部的已知缺陷说明)。 |
| `utils/neural_network/` | GraphGRU 预测管线的模型/训练/预处理,仅被 `run_prediction_*` 使用。 |

### 数据与产物(均已 gitignore)

- `data/` — 图 pkl、几何 CSV、SLD/ACS 缓存(`space_function/`、`socioeconomic/`)、
  疏散命令数据层(`evacuation_orders/`,自带 README)。
- `outputs/` — 各管线产物,子目录互不相交:`nmf/`(生产)与 `archive/`(归档脚本的产物:
  `tucker/`、`temporal_decay/`、`distance_decay_paired/` 及两个结果 pkl)。
- 图 pkl 的生成不在本仓库,见 `notebook_drafts/Cubique/main_20260627.ipynb`。

### 文档

`docs/technical_notes/` 三篇技术笔记(成分特征探索、韧性回归、跨城市预测)。笔记用
`file.py:行号` 锚点指向源码;改动这些文件的行号后需同步修订笔记(历史上均已同步)。

## 术语表

### 城市-事件的四个标识符(`CITY_EVENTS` 注册表,run_pattern_nmf.py)

| 名称 | 示例 | 用途 |
|---|---|---|
| `code` | `BR_Ida` | 城市-事件的主键:跨城市 split、feats_by_city 的键、图表列名 |
| `label` | `Baton Rouge` | 人类可读城市名(打印、图题、load_city_geo 的报错) |
| `key` | `Baton_Rouge` | 文件名前缀(SLD/ACS 缓存、几何 CSV) |
| `tag` | `_BR_Ida` | 输出文件名后缀(每单元的图/CSV 命名) |

同一城市可对应多个事件(Wilmington 有 `WM_Dorian` 与 `WM_Isaias`,共享 `label`/`key`/几何,
`code` 不同)。循环变量约定:`u` 指 main() 组装的单元 dict(含 W/H/gdf 等),`cfg` 指
注册表原始条目;两者都以 `code` 为键。

### 地理单元标识符

| 名称 | 示例 | 出处 |
|---|---|---|
| `aggr_id` | `US.LA.033.004006.3` | Snowflake geography_registry 导出;分析侧主键(geo CSV 的 `geography_id` 列在 geo_loader 中重命名而来) |
| `geoid20` / `GEOID20` | `220330040063` | 12 位 Census FIPS;`aggr_id_to_geoid20`(fetch_sld_landuse.py)做转换;大写形式是外部 API 字段名 |
| `fips_code` | `220330040063` | 同一 geo CSV 中与 `geography_id` 并存的 FIPS 列;疏散数据层(data/evacuation_orders)以它为键 |

即:同一张 `<key>_block_group_geo.csv` 同时携带 `geography_id`(→`aggr_id`)与
`fips_code` 两种键,生产管线用前者,疏散层用后者,经 `aggr_id_to_geoid20` 可互换。

### 核心矩阵与符号

| 符号 | 含义 |
|---|---|
| `X_all` | 一个城市-事件的 [OD 对 × 时间槽] 流量矩阵(全窗口) |
| `n_nor` / `n_dis` | 列索引分界:`[0,n_nor)` 正常期,`[n_nor,n_dis)` 缓冲期,`[n_dis,…)` 灾害期 |
| `W` | NMF 时间因子 [时间槽 × k],列 L2 归一 |
| `H` | NMF 空间因子 [k × OD 对],吸收幅度 |
| `M` | O×D 功能交叉表 [k × C × C](space_function.build_od_function_matrix) |
| `weights` | 成分重要度 ‖W列‖·‖H行‖(全窗口) |
| `weight_normal` | 成分在正常期基线中的份额(城市级 cum_loss 重建的聚合权重,≠ `weights`) |
| `feats` | 逐成分特征表(一行一个 NMF 成分,含预测子列 + 韧性目标列) |

## 运行环境

conda 环境 `disaster_mobi`(`C:\Users\xp2239\.conda\envs\disaster_mobi\python.exe`),
Windows 下建议设 `PYTHONUTF8=1`。Census API key 放在
`data/socioeconomic/census_api_key.txt`(SLD/ACS 缓存已就位时不需要)。
