# 充电站错配指数 M(F) 

## 结构
```
ChargingStationOptimization/
├── data/                              原始数据（轨迹 parquet + 站点 csv）
├── Code/
│   ├── cso.py                         单文件引擎：参数 + 数据 + 需求面 + 三指标 + 容量感知选站
│   ├── run_all.py                     多城市一键跑两口径×两指标×S1/S2/S3，落盘 Outputs/<city>/<mode>/
│   ├── STEP_1_data_explore.ipynb      分阶段：数据探索
│   ├── STEP_2_soc_demand.ipynb        分阶段：轨迹→SoC→需求面
│   ├── STEP_3_baseline_scenarios.ipynb 分阶段：基线三指标 + 容量感知 S1/S2/S3
│   └── STEP_4_confidence_sensitivity.ipynb 分阶段：参数敏感性
├── Outputs/<city>/<mode>/             run_all 产物（baseline/s1/s2/s3/cross_score + 图）
├── 错配指数M改进_方法论.md
└── _archive/                          旧代码（已废弃）
```

## 三个指标（方法论 §1/§5）
- `M_old`（式1）：无限容量、最近站硬指派，**仅作零拥挤极限参照**。
- `M_disp` 空间置换（式3-4）：容量受限运输 LP（scipy HiGHS 精确解），拥堵=绕路，单位 需求·km。偏乐观。
- `M_queue` 时间排队（式5-7）：最近站指派 + M/M/c Erlang-C 等待，拥堵=排队，单位 需求·min。偏悲观。

S1 只增 / S2 只减 / S3 等量调配 **直接在容量口径上选站**（两口径各一遍）；空间置换 LP 的容量标尺 `s` 在选站时固定在基线站集（避免增删站改写幸存站容量）。

## 运行
```bash
pip install numpy pandas polars pyarrow scipy matplotlib   # 依赖
cd Code
python run_all.py                  # 正式精度，跑 truncated + comprehensive
SMOKE=1 python run_all.py          # 冒烟：小抽样小规模先把链路走通
python run_all.py truncated        # 只跑某一口径
python run_all.py --city guangzhou truncated comprehensive
# 分阶段：打开 STEP_1..4，把首格 MODE 改 'truncated'/'comprehensive'
```
首次运行会读轨迹 parquet 并缓存到 `data/_segments_cache_<city>.npz`；当前广州旧缓存 `data/_segments_cache.npz` 仍可兼容读取。所有可调参数在 `cso.py` 顶部。

## 多城市配置
默认城市是 `guangzhou`。全国多城市建议把每个城市的数据放在 `data/<city>/` 下，并用外部 JSON 提供文件名、bbox 和行政区 adcode；bbox 是经纬度裁剪框，必须逐城设置，否则轨迹和站点会被错误裁剪。

`city_configs.json` 示例：
```json
{
  "cities": {
    "shenzhen": {
      "name_cn": "深圳",
      "trace_file": "Taxi_2019_10_14_admin_4403.parquet",
      "station_file": "shenzhen_station.csv",
      "trace_date": "2019-10-14",
      "bbox": {"lon_min": 113.70, "lon_max": 114.65, "lat_min": 22.35, "lat_max": 22.90},
      "admin_adcode": "440300",
      "station_cols": {
        "lon": "WGS84_station_lg",
        "lat": "WGS84_station_lt",
        "fast": "station_fast_cnt",
        "slow": "station_slow_cnt",
        "create_time": "create_time",
        "sid": "station_id"
      }
    }
  }
}
```

运行：
```bash
python run_all.py --city shenzhen --city-config ../city_configs.json
```
