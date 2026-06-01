# Code/ — 充电站布局优化 · 分阶段分析

| 笔记本 | 内容 |
|---|---|
| `STEP_1_data_explore` | 数据探索：规模、范围、采样/速度分布、站点供给 |
| `STEP_2_soc_demand` | 轨迹→SoC→低电量需求；**对初始电量做蒙特卡洛集成**得期望需求面 |
| `STEP_3_baseline_scenarios` | 基线错配 M₀ + S1 只增 / S2 只减(鸽笼分解) / S3 等量调配 |
| `STEP_5_conclusions` | 假设判定、讨论、局限、**多城市扩展路线** |

**依赖**：`numpy pandas polars matplotlib jupyter`。
**数据**：`Taxi_*.parquet` 与 `*_station.csv` 默认放在本 `Code/` 目录的**上一级**（项目根）；或设环境变量 `CSO_DATA` 指向数据目录。
**换城市**：见  `cso_config.py` 的 `CITIES` 注册表——加一条配置 + `C.use_city('name')`。

### 一键运行

**批量脚本 `Code/run_all_cities.py`**：遍历 `CITIES` 里填好的每个城市，在 `Figure/<city>_<date>/` 下生成 STEP_1~3 的全部 13 张图 + 6 个分析文件（`metrics.json`、`summary.txt`、`station_filter.csv`、`district_*.csv`、`s1/s3` 表）。

- 跑所有城市 

```python
python run_all_cities.py
```

- 指定城市 

```python
python run_all_cities.py --cities guangzhou 
```

- 数据/日期变更后强制重算缓存

```python
python run_all_cities.py --rebuild
```

