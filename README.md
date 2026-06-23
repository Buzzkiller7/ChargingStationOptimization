# 根据网约车轨迹优化充电站 · 2SFCA 错配指数 M(F)

用出租车/网约车轨迹估计「期望低电量需求面」，再用 2SFCA 可达性框架衡量某一年充电站存量下的供需错配 `M`，并给出 S1/S2/S3 三类布局调整与「模型推荐 vs 真实下一年新增」的对照。

## 目录结构

```text
ChargingStationOptimization/
├── data/                                 原始数据与缓存
│   ├── Taxi_2019_10_14..20_admin_4401.parquet   多日轨迹
│   ├── guangzhou_station.csv             站点表
│   ├── admin_440100.json                 行政边界
│   ├── _graph_guangzhou.graphml          OSMnx 路网缓存（首次联网构图后落盘）
│   └── _segments_cache_*.npz             逐日轨迹分段缓存
├── Code/
│   ├── cso.py                            配置/数据/路网/需求/指标/优化/年度/绘图/冒烟
│   ├── style.py                          统一字体配色 + 路网/需求路段着色 + savefig
│   ├── run_all.py                        批处理：逐年单独出成果
│   ├── STEP_1.ipynb                      数据审计
│   ├── STEP_2.ipynb                      SoC 与固定需求面 + 现状错配
│   ├── STEP_3.ipynb                      年度 S1/S2/S3 + 真实新增对照
│   └── STEP_4.ipynb                      六参数敏感性
├── Outputs/<city>/                       run_all 产物
│   └── yearly/<Y>/                       每个年份单独
├── 充电站错配指数 M(F).md                方法说明
└── LICENSE
```

## 指标定义（2SFCA 族）

错配指数分三项，单位都是「需求·km」：

```text
M = M_access + M_crowd + M_reach
```

`M_access` 是 G2SFCA 高斯引力分配下的期望出行距离（需求侧可达性，只在同一 `d0` 口径下做相对比较）；`M_crowd` 是 i2SFCA 拥挤项，用站点拥挤度 `C_j = L_j / b_j`（`L_j` 为 2SFCA 分配负载、`b_j = s·κ_j` 为容量标尺）超过 1 的溢出比例折算成 km 当量；`M_reach` 是「够不着」惩罚，把每一次抛锚按政策权重 `P_DEAD` 折算成出行 km。有效容量 `κ_j = γ·n_fast + n_slow`。

## 运行

需要的依赖：

```bash
pip install numpy pandas polars pyarrow scipy networkx osmnx pyproj shapely matplotlib tqdm
```

批处理入口 `run_all.py`，**每个年份单独出成果**到 `Outputs/<city>/yearly/<Y>/`：

```bash
cd Code
SMOKE=1 python run_all.py                 # 冒烟：单日少抽样、小候选，Outputs/_smoke，不覆盖正式结果
python run_all.py                         # 正式：全年份、S1/S2/S3 全跑
python run_all.py --years 2019,2021,2023  # 只跑指定年份
python run_all.py --strategies s1         # 只跑某些策略（s1/s2/s3 任意子集）
python run_all.py --backend fast          # CELF 后端 fast（默认 dense，全比例下更省内存）
python run_all.py --city guangzhou
```

## 多城市配置

默认城市 `guangzhou`。换城市时把数据放在 `data/<city>/`（或 `data/` 下按通配命名），并用外部 JSON 提供文件名通配、坐标基准、bbox、行政区 adcode 与站点字段名；bbox 是经纬度裁剪框，必须逐城设置，否则轨迹和站点会被错误裁剪。

`city_configs.json` 示例：

```json
{
  "cities": {
    "shenzhen": {
      "name_cn": "深圳",
      "trace_glob": "Taxi_*_admin_4403.parquet",
      "trace_date_re": "Taxi_(\\d{4}_\\d{2}_\\d{2})_admin",
      "trace_datum": "wgs84",
      "station_file": "shenzhen_station.csv",
      "station_datum": "wgs84",
      "bbox": {"lon_min": 113.70, "lon_max": 114.65, "lat_min": 22.35, "lat_max": 22.90},
      "metric_epsg": 32649,
      "admin_adcode": "440300",
      "station_cols": {
        "lon": "WGS84_station_lg", "lat": "WGS84_station_lt",
        "fast": "station_fast_cnt", "slow": "station_slow_cnt",
        "create_time": "create_time", "sid": "station_id"
      }
    }
  }
}
```

```bash
python run_all.py --city shenzhen --city-config ../city_configs.json
```
