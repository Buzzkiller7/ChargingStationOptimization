# -*- coding: utf-8 -*-
"""
cso.py — 充电站错配指数 M(F) 的单文件计算引擎（2SFCA 重构版）。

整条链路：
  轨迹 GPS → 逐段能耗 → 电量 SoC → 低电量需求面 w_i（多日池化，映射到路网节点中心）
  → 需求点到站点距离 c_ij（OSMnx 路网最短路，无 haversine/网格直线近似）
  → 2SFCA 错配指数 M(F) = M_access(期望出行) + M_crowd(i2SFCA 拥挤) + M_reach(够不着)
  → S1 只增 / S2 只减 / S3 等量调配（贪心 + CELF，目标=新指标）

核心方法：G2SFCA 高斯衰减引力分配（需求侧可达性）+ i2SFCA 拥挤度（供给侧竞争），
只用“容量 κ / 需求 w / 距离”，无任何行为参数；详见《错配指数M_2SFCA重构_方法论_20260621.md》。
约定：距离 km；M 单位 需求·km。所有公式已在真实广州数据上端到端验证（退化到 M_old 误差 0.00%）。
"""
from __future__ import annotations
import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

try:                                                  
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **k): return it

# ============================================================================
# 0. 参数
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent          # 项目根（Code 的上一级）
BASE_DATA = ROOT / "data"

# 城市配置
CITY_CONFIGS = {
    "guangzhou": {
        "name_cn": "广州",
        "trace_glob": "Taxi_*_admin_4401.parquet",          # 多日轨迹文件通配
        "trace_date_re": r"Taxi_(\d{4}_\d{2}_\d{2})_admin",  # 从文件名解析日期 YYYY_MM_DD
        "trace_datum": "wgs84",                              # 轨迹坐标基准；若为 GCJ-02 改 "gcj02"
        "station_file": "guangzhou_station.csv",
        "station_datum": "wgs84",
        "bbox": dict(lon_min=112.90, lon_max=114.10, lat_min=22.50, lat_max=24.00),
        "metric_epsg": 32649,                               # 米制投影 UTM 49N（snap/欧氏用）
        "admin_adcode": "440100",
        "station_cols": dict(lon="WGS84_station_lg", lat="WGS84_station_lt",
                             fast="station_fast_cnt", slow="station_slow_cnt",
                             create_time="create_time", sid="station_id"),
    }
}

# —— 由 configure_city 填充的当前城市状态 ——
CITY = "guangzhou"; CITY_NAME = "广州"; CITY_ADMIN_ADCODE = "440100"
DATA = BASE_DATA; BBOX = {}; STATION_COLS = {}
CITY_CFG = {}; DAYS = []; TRACE_CUTOFF = ""             # DAYS=[(date_str, Path), ...]

# 车型能耗（时间用小时）：dE = d·(k_d + k_v2·v²) + dt·k_t
BATT = 78.4; K_D = 0.150; K_V2 = 0.000025; K_T = 0.9

# 需求定义
SOC_LOW = 0.20; SEG_DT_CLIP = 0.5; SEG_D_CLIP = 50.0; MIN_TRACK_POINTS = 100
SOC0 = dict(mean=0.85, sd=0.10, lo=0.40, hi=1.00)

# 成本 / 可达
C_BAR = 100.0                    # 低电量可达里程 km（判定可达/抛锚）
NET_CUTOFF_KM = C_BAR            # 路网最短路 Dijkstra 截断半径(km)：必须覆盖 C_BAR，不能按 d0 截短
P_DEAD = 1000.0                  # 抛锚惩罚（km 当量）
SHORTEST_PATH_BACKEND = os.environ.get("CSO_SHORTEST_PATH_BACKEND", "scipy").strip().lower()
DIJKSTRA_BATCH = int(os.environ.get("CSO_DIJKSTRA_BATCH", "64"))
# S1/S3 贪心加站内层评估后端：'fast'=增量聚合（默认，快几十倍），'dense'=逐候选 concat+整重算（参照/对拍用）。
# 两者数值等价（已在合成数据上对拍到机器精度），结果与选站序列一致；'dense' 仅用于交叉验证。
GREEDY_BACKEND = os.environ.get("CSO_GREEDY_BACKEND", "fast").strip().lower()

# 容量与 2SFCA 核心
GAMMA = 10.0                     # 快/慢桩有效容量比 κ=γ·fast+slow
SYS_UTIL = 1.0                   # 系统利用率 u（总需求/总容量），定容量标尺 s；扫 0.8-1.5
D0_DECAY = 8.0                   # 高斯距离衰减带宽 km（核心参数；扫 3-15）
BETA_CROWD = 1.0                 # 拥挤代价系数（扫 0.5-2）

# 集成 / 选站
N_ENSEMBLE = 40                  # 初始 SoC 蒙特卡洛抽样数
# 三类策略各用各的比例，规模 = 现有站数 × 比例。
ADD_FRACS    = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)               # S1 只增
REMOVE_FRACS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)  # S2 只减
SWAP_FRACS   = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)               # S3 等量调配
CAND_CAP = 1200; CAP_POOL = 200; SEED = 25

# 距离一律用 OSMnx 路网最短路
DIST_BACKEND = "network"

# —— 年度分析（固定需求面 + 逐年站点存量）相关参数 ——
# 需求面年份：None=用发现到的轨迹年（当前广州=2019）；填整数则显式指定需求年（与存量年 Y 解耦）。
DEMAND_YEAR = None
# 年度存量口径："cumulative"=截至 Y 年底（create_time <= Y-12-31）的全部站，符合“逐年累计存量”。
YEARLY_MODE = "cumulative"
# 真实新增对照的命中阈值(km)：真实新增站到最近推荐点 <= 阈值算命中（用于阈值命中率曲线）。
MATCH_THRESH_KM = (1.0, 2.0, 5.0)
OUTPUT_ROOT = ROOT / "Outputs"   # 输出根目录集中管理；notebook/冒烟一律写临时子目录，不覆盖正式 Outputs

# —— STEP_4 参数敏感性默认扫描网格——
# 每次只动一个参数、其余固定；P_DEAD 行报告时务必同时给原生可达覆盖率（它是政策权重，不是测量出的公里）。
SENS_GRID = {
    "d0":     (3, 5, 8, 12, 15),       # 高斯衰减带宽 km
    "gamma":  (5, 8, 10, 15),          # 快/慢桩有效容量比 γ
    "u":      (0.8, 1.0, 1.2, 1.5),    # 系统压力情景 u（非物理日利用率）
    "beta":   (0.5, 1.0, 1.5, 2.0),    # 拥挤代价系数 β
    "C_BAR":  (60, 80, 100, 120),      # 低电量物理可达半径 km
    "P_DEAD": (250, 500, 1000, 2000),  # 抛锚惩罚（km 当量，政策权重）
}

# 统一冒烟开关：SMOKE=1 时减少抽样/天数/候选规模，仅用于快速跑通链路，不改任何参数语义。
SMOKE = os.environ.get("SMOKE", "").strip().lower() in {"1", "true", "yes"}


# ============================================================================
# 1. 城市配置 + 多日发现
# ============================================================================
def _load_city_configs(config_path=None):
    configs = dict(CITY_CONFIGS)
    path = config_path or os.environ.get("CSO_CITY_CONFIG", "").strip()
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            extra = json.load(f)
        extra = extra.get("cities", extra)
        for k, v in extra.items():
            configs[str(k).lower()] = v
    return configs


def discover_days(cfg, data_dir):
    """按 trace_glob 发现实际存在的日文件，解析日期升序返回 [(date_str, Path), ...]。
    天数 = len(返回)，完全由数据决定，代码不写死。"""
    rex = re.compile(cfg["trace_date_re"]); out = []
    for p in sorted(Path(data_dir).glob(cfg["trace_glob"])):
        m = rex.search(p.name)
        if m:
            out.append((m.group(1), p))
    out.sort(key=lambda t: t[0])
    if not out:
        raise FileNotFoundError(f"在 {data_dir} 未发现匹配 {cfg['trace_glob']} 的轨迹文件")
    return out


def configure_city(city=None, config_path=None):
    """切换当前城市配置：影响轨迹文件、站点文件、bbox、行政区 adcode、多日列表与输出标签。"""
    global CITY, CITY_NAME, CITY_ADMIN_ADCODE, DATA, BBOX, STATION_COLS, CITY_CFG, DAYS, TRACE_CUTOFF
    global _GRAPH, _NODE_LL, _GRAPH_CSR_CACHE
    configs = _load_city_configs(config_path)
    key = (city or os.environ.get("CSO_CITY", CITY) or "guangzhou").strip().lower()
    if key not in configs:
        raise KeyError(f"未知城市：{key}；可用 {sorted(configs)}")
    cfg = configs[key]
    CITY = key
    CITY_NAME = str(cfg.get("name_cn") or cfg.get("name") or key)
    CITY_ADMIN_ADCODE = str(cfg.get("admin_adcode", "") or "")
    BBOX = dict(cfg["bbox"]); STATION_COLS = dict(cfg["station_cols"]); CITY_CFG = cfg
    city_data = BASE_DATA / CITY
    DATA = city_data if city_data.exists() else BASE_DATA
    DAYS = discover_days(cfg, DATA)
    TRACE_CUTOFF = min(d for d, _ in DAYS).replace("_", "-")   # 观测期开始 → truncated 因果切点
    _GRAPH = None; _NODE_LL = None; _GRAPH_CSR_CACHE = None   # 切城市清掉路网/CSR缓存
    return cfg


# ============================================================================
# 2. 几何 / 坐标系 / 栅格
# ============================================================================
def haversine_km(lon1, lat1, lon2, lat2):
    """经纬度球面距离 (km)，支持 numpy 广播。"""
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(d))


def _gcj02_to_wgs84(lon, lat):
    """GCJ-02（火星坐标）→ WGS84 标准纠偏。国内商用 GPS 常为 GCJ-02，与 WGS84 路网混用会错位。"""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    a = 6378245.0; ee = 0.00669342162296594323
    x = lon - 105.0; y = lat - 35.0
    dlat = (-100 + 2*x + 3*y + 0.2*y*y + 0.1*x*y + 0.2*np.sqrt(np.abs(x))
            + (20*np.sin(6*x*np.pi) + 20*np.sin(2*x*np.pi)) * 2/3
            + (20*np.sin(y*np.pi) + 40*np.sin(y/3*np.pi)) * 2/3
            + (160*np.sin(y/12*np.pi) + 320*np.sin(y*np.pi/30.0)) * 2/3)
    dlon = (300 + x + 2*y + 0.1*x*x + 0.1*x*y + 0.1*np.sqrt(np.abs(x))
            + (20*np.sin(6*x*np.pi) + 20*np.sin(2*x*np.pi)) * 2/3
            + (20*np.sin(x*np.pi) + 40*np.sin(x/3*np.pi)) * 2/3
            + (150*np.sin(x/12*np.pi) + 300*np.sin(x/30.0*np.pi)) * 2/3)
    radlat = lat / 180.0 * np.pi
    magic = 1 - ee * np.sin(radlat) ** 2; sqm = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqm) * np.pi)
    dlon = (dlon * 180.0) / (a / sqm * np.cos(radlat) * np.pi)
    return lon - dlon, lat - dlat


def to_wgs84(lon, lat, datum):
    if str(datum).lower() in ("gcj02", "gcj-02", "mars"):
        return _gcj02_to_wgs84(lon, lat)
    return np.asarray(lon, float), np.asarray(lat, float)


# ============================================================================
# 3. 数据加载（站点 + 多日轨迹）
# ============================================================================
def _load_stations_raw():
    """读站点 CSV → 坐标统一到 WGS84、解析容量与建成时间，返回**全部原始行**的标准化表。
    列：lon, lat, fast, slow, sid(str), create_time(datetime64，解析失败为 NaT),
        geo_ok(bool：坐标有限 & 落在 bbox 内 & 有效容量(fast+slow)>0)。
    这是 load_stations / station_snapshot / station_years / real_additions_between 的**共用过滤核**——
    年度切片与 truncated/comprehensive 口径必须同源，避免两套过滤逻辑漂移。"""
    sc = STATION_COLS; b = BBOX
    raw = pd.read_csv(DATA / CITY_CFG["station_file"])
    lon = pd.to_numeric(raw[sc["lon"]], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(raw[sc["lat"]], errors="coerce").to_numpy(float)
    lon, lat = to_wgs84(lon, lat, CITY_CFG.get("station_datum", "wgs84"))
    fast = pd.to_numeric(raw[sc["fast"]], errors="coerce").fillna(0).to_numpy(float)
    slow = pd.to_numeric(raw[sc["slow"]], errors="coerce").fillna(0).to_numpy(float)
    created = pd.to_datetime(raw[sc["create_time"]], errors="coerce")
    geo_ok = (np.isfinite(lon) & np.isfinite(lat)
              & (lon >= b["lon_min"]) & (lon <= b["lon_max"])
              & (lat >= b["lat_min"]) & (lat <= b["lat_max"]) & ((fast + slow) > 0))
    return pd.DataFrame(dict(lon=lon, lat=lat, fast=fast, slow=slow,
                             sid=raw[sc["sid"]].astype(str).to_numpy(),
                             create_time=created.to_numpy(), geo_ok=geo_ok))


def load_stations(mode="truncated", return_stats=False):
    """读站点表，保留 bbox 内、坐标有效、容量>0 的站。坐标按 station_datum 统一到 WGS84。
    mode='truncated'：只保留观测期开始(TRACE_CUTOFF)前已建成的站；'comprehensive'：全部现存站。
    （对外行为与重构前完全一致；底层改为复用 _load_stations_raw 的过滤核，已用新旧对拍验证。）"""
    full = _load_stations_raw()
    ok = full["geo_ok"].to_numpy().copy()
    created = pd.to_datetime(full["create_time"])
    fast = full["fast"].to_numpy(); slow = full["slow"].to_numpy()
    n_future = 0
    if mode == "truncated":
        cutoff = pd.Timestamp(TRACE_CUTOFF)
        date_ok = created.notna() & (created.dt.normalize() <= cutoff)
        n_future = int((created.dt.normalize() > cutoff).sum())   # 晚于切点的站数（口径同旧版，覆盖全部原始行）
        ok = ok & date_ok.to_numpy()
    st = full.loc[ok, ["lon", "lat", "fast", "slow", "sid"]].reset_index(drop=True)
    stats = dict(mode=mode, raw=int(len(full)), kept=int(len(st)),
                 dropped=int(len(full) - len(st)), future_create=n_future,
                 zero_cap=int(((fast + slow) <= 0).sum()))
    return (st, stats) if return_stats else st


def station_snapshot(year, return_stats=False):
    """年度存量切片：返回截至 {year} 年底（create_time <= {year}-12-31）、且通过 bbox+容量过滤的站点存量。
    与 load_stations 共用 _load_stations_raw 过滤核；用于“固定需求面 + 逐年供给演化”的年度分析。
    列与 load_stations 一致：lon, lat, fast, slow, sid。"""
    full = _load_stations_raw()
    cutoff = pd.Timestamp(f"{int(year)}-12-31")
    created = pd.to_datetime(full["create_time"])
    sel = (full["geo_ok"].to_numpy()
           & created.notna().to_numpy()
           & (created.dt.normalize() <= cutoff).to_numpy())
    st = full.loc[sel, ["lon", "lat", "fast", "slow", "sid"]].reset_index(drop=True)
    if not return_stats:
        return st
    return st, dict(year=int(year), kept=int(len(st)), raw=int(len(full)), cutoff=str(cutoff.date()))


def station_years():
    """从站点 create_time 自动发现年份范围，返回 [最早年 … 最晚年] 的连续整数列表（不手写年份）。
    只统计通过 bbox+容量过滤的有效站，使年份反映“可分析存量”的时间跨度。"""
    full = _load_stations_raw()
    yrs = pd.to_datetime(full.loc[full["geo_ok"], "create_time"]).dt.year.dropna()
    if yrs.empty:
        raise ValueError("站点 create_time 无法解析出任何年份，无法做年度分析")
    return list(range(int(yrs.min()), int(yrs.max()) + 1))


def real_additions_between(year, year_next):
    """{year}→{year_next} 之间真实新增的站：create_time ∈ ({year}-12-31, {year_next}-12-31]。
    **仅用于与模型推荐做对照评估，绝不参与推荐计算（防时间泄漏）。**
    列与 station_snapshot 一致：lon, lat, fast, slow, sid。"""
    full = _load_stations_raw()
    lo = pd.Timestamp(f"{int(year)}-12-31"); hi = pd.Timestamp(f"{int(year_next)}-12-31")
    created = pd.to_datetime(full["create_time"]); norm = created.dt.normalize()
    sel = (full["geo_ok"].to_numpy()
           & created.notna().to_numpy()
           & (norm > lo).to_numpy() & (norm <= hi).to_numpy())
    return full.loc[sel, ["lon", "lat", "fast", "slow", "sid"]].reset_index(drop=True)


def _segments_from_parquet(path):
    """从单日 parquet 算相邻点 距离d(km)/时间dt(h)/速度v(km/h)。坐标按 trace_datum 统一到 WGS84。"""
    import polars as pl
    b = BBOX
    lf = (pl.scan_parquet(str(path))
          .select(["vehicle_id", pl.col("lon").cast(pl.Float64), pl.col("lat").cast(pl.Float64),
                   pl.col("speed_kmh").cast(pl.Float64), "gps_time"])
          .filter(pl.col("lon").is_between(b["lon_min"], b["lon_max"]) &
                  pl.col("lat").is_between(b["lat_min"], b["lat_max"]))
          .with_columns((pl.col("gps_time").str.slice(11, 2).cast(pl.Int32, strict=False) * 3600
                         + pl.col("gps_time").str.slice(14, 2).cast(pl.Int32, strict=False) * 60
                         + pl.col("gps_time").str.slice(17, 2).cast(pl.Int32, strict=False)).alias("t"))
          .filter(pl.col("t").is_not_null() & (pl.col("t") >= 0))
          .filter(pl.col("lon").count().over("vehicle_id") >= MIN_TRACK_POINTS)
          .with_columns(pl.col("vehicle_id").cast(pl.String).cast(pl.Categorical).to_physical().alias("vc"))
          .sort(["vc", "t"]).select(["vc", "lon", "lat", "speed_kmh", "t"]))
    df = lf.collect()
    vc = df["vc"].to_numpy().astype(np.int32)
    lon = df["lon"].to_numpy(); lat = df["lat"].to_numpy()
    lon, lat = to_wgs84(lon, lat, CITY_CFG.get("trace_datum", "wgs84"))
    spd = df["speed_kmh"].to_numpy(); t = df["t"].to_numpy().astype(np.float64)
    n = len(vc)
    start = np.empty(n, bool); start[0] = True; start[1:] = vc[1:] != vc[:-1]
    d = np.zeros(n); dt = np.zeros(n); v = np.zeros(n)
    d[1:] = haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dt[1:] = (t[1:] - t[:-1]) / 3600.0
    v[1:] = 0.5 * (spd[1:] + spd[:-1])
    d[start] = 0; dt[start] = 0; v[start] = 0
    np.clip(d, 0, SEG_D_CLIP, out=d); np.clip(dt, 0, SEG_DT_CLIP, out=dt)
    return dict(vc=vc.astype(np.int32), lon=lon.astype(np.float32), lat=lat.astype(np.float32),
                d=d.astype(np.float32), dt=dt.astype(np.float32), v=v.astype(np.float32),
                start=start, n_veh=np.int64(vc.max() + 1))


def load_segments(date_str=None, path=None):
    """读某一日 segments（按天缓存）。date_str/path 缺省取 DAYS[0]。广州 2019_10_14 兼容旧 legacy 缓存。"""
    if date_str is None:
        date_str, path = DAYS[0]
    cache = DATA / f"_segments_cache_{CITY}_{date_str}.npz"
    legacy = DATA / "_segments_cache.npz"
    src = cache if cache.exists() else (legacy if (CITY == "guangzhou" and date_str == "2019_10_14" and legacy.exists()) else None)
    if src is not None:
        try:
            z = np.load(src, allow_pickle=False); out = {k: z[k] for k in z.files}; z.close(); return out
        except Exception:
            pass
    print(f"[数据] 读取 {date_str} 轨迹 parquet（首次较慢，之后走缓存）...", flush=True)
    out = _segments_from_parquet(path)
    try:
        np.savez(cache, **out)
    except Exception:
        pass
    return out


# ============================================================================
# 4. 能耗 → SoC → 低电量需求面（多日池化，映射到路网节点中心）
# ============================================================================
def cumulative_kwh(d, dt, v, start):
    seg = d * (K_D + K_V2 * v * v) + dt * K_T
    csum = np.cumsum(seg)
    reset = np.zeros(len(seg)); idx = np.where(start)[0]
    reset[idx[1:]] = csum[idx[1:] - 1]
    return csum - np.maximum.accumulate(reset)


def low_soc_events(cum, vc, lon, lat, soc0_per_veh):
    ce = cum + ((1.0 - soc0_per_veh) * BATT)[vc]
    soc = 1.0 - (ce - np.floor(ce / BATT) * BATT) / BATT
    cyc = np.floor(ce / BATT).astype(np.int64)
    m = soc <= SOC_LOW
    key = (vc.astype(np.int64) * 100003 + cyc)[m]
    _, first = np.unique(key, return_index=True)
    return lon[m][first], lat[m][first]


def example_soc_curves(percentiles=(30, 50, 70, 90), soc0=None):
    """取若干辆代表车的 SoC 曲线示例（用 DAYS[0]）。返回 list[dict(veh, pct, cum_km, soc)]。"""
    z = load_segments(); vc, d = z["vc"], z["d"].astype(float)
    cum = cumulative_kwh(z["d"], z["dt"], z["v"], z["start"])
    n_veh = int(z["n_veh"]); last = np.zeros(n_veh); np.maximum.at(last, vc, cum)
    soc0 = SOC0["mean"] if soc0 is None else soc0
    pos = last > 0; out = []
    for p in percentiles:
        thr = np.percentile(last[pos], p)
        vv = int(np.where(pos)[0][np.argmin(np.abs(last[pos] - thr))])
        m = vc == vv; ce = cum[m] + (1.0 - soc0) * BATT
        soc = 1.0 - (ce - np.floor(ce / BATT) * BATT) / BATT
        out.append(dict(veh=vv, pct=p, cum_km=np.cumsum(d[m]), soc=soc))
    return out


def demand_surface(draws=None, seed0=1000, days=None):
    """多日池化需求面：逐日做 SoC0 蒙特卡洛、取每(车,周期)首次低电量点，
    **一律 snap 到最近路网节点中心**再聚合，跨所有天与抽样平均得到期望日需求量 w_i。天数由数据决定。
    需求只走路网节点中心——拿不到路网图就直接报错，**不回退 0.01° 网格**。
    返回 dict(lon_c, lat_c, w, n_cells, n_days, space='node')。"""
    draws = N_ENSEMBLE if draws is None else int(draws)
    days = DAYS if days is None else days
    n_days = len(days)
    if not _osmnx_ok():
        raise RuntimeError("需求必须映射到路网节点中心，但未安装 osmnx。请先 `pip install osmnx` 再运行。")
    try:
        _get_graph()                       # 构建/载入路网图并捕获节点经纬度（失败即报错，不回退网格）
    except Exception as e:
        raise RuntimeError(
            f"需求必须映射到路网节点中心，但路网图不可用：{type(e).__name__}: {e}。"
            f"请联网首次构图；若缓存损坏/版本不符，删除 data/_graph_{CITY}.graphml 后重试。") from e
    blocks = []
    _desc = f"需求面·{len(days)}天×{draws}抽样(路网节点中心)"
    for di, (date_str, path) in enumerate(tqdm(days, desc=_desc)):
        z = load_segments(date_str, path)
        vc, lon, lat, n_veh = z["vc"], z["lon"], z["lat"], int(z["n_veh"])
        cum = cumulative_kwh(z["d"], z["dt"], z["v"], z["start"])
        for m in range(draws):
            rng = np.random.default_rng(seed0 + di * 100000 + m)
            s0 = np.clip(rng.normal(SOC0["mean"], SOC0["sd"], n_veh), SOC0["lo"], SOC0["hi"])
            el, ea = low_soc_events(cum, vc, lon, lat, s0)
            ub, cnt = np.unique(_events_to_nodes(el, ea), return_counts=True)
            blocks.append((ub, cnt.astype(float)))
    master = np.unique(np.concatenate([u for u, _ in blocks]))
    w = np.zeros(len(master))
    for ub, cnt in blocks:
        w[np.searchsorted(master, ub)] += cnt
    w /= (draws * n_days)
    lc = np.array([_NODE_LL[int(n)][0] for n in master])
    ac = np.array([_NODE_LL[int(n)][1] for n in master])
    return dict(lon_c=lc, lat_c=ac, w=w, node_id=master.astype(np.int64),
                n_cells=len(master), n_days=n_days, space="node")


# ============================================================================
# 5. 路网（OSMnx）：构图 / 节点 snap / 最短路；可回退 haversine
# ============================================================================
_GRAPH = None      # OSMnx 投影图缓存（进程内，米制，用于 snap/最短路）
_NODE_LL = None    # {node_id: (lon, lat)} WGS84 节点坐标（投影前捕获，用于需求节点中心）
_GRAPH_CSR_CACHE = None    # scipy 稀疏图缓存：(id(G), nodes, node_to_idx, csr_reverse)


# —— OSMnx 跨版本兼容封装（1.x 与 2.x 的 API 位置/签名不同；避免因版本不一致直接报错）——
def _ox_graph_from_bbox(ox, b):
    """2.x：graph_from_bbox(bbox=(left,bottom,right,top))；1.x：graph_from_bbox(north,south,east,west)。"""
    try:
        return ox.graph_from_bbox(bbox=(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"]),
                                  network_type="drive")
    except TypeError:
        return ox.graph_from_bbox(b["lat_max"], b["lat_min"], b["lon_max"], b["lon_min"],
                                  network_type="drive")


def _ox_add_traveltimes(ox, G):
    """2.x：ox.routing.add_edge_*；1.x：ox.add_edge_*。失败则跳过（仅影响可选的时间口径）。"""
    for mod in (getattr(ox, "routing", None), ox):
        try:
            G = mod.add_edge_speeds(G); G = mod.add_edge_travel_times(G); return G
        except Exception:
            continue
    return G


def _ox_nearest_nodes(ox, G, X, Y):
    """1.x：ox.distance.nearest_nodes；2.x：ox.nearest_nodes。"""
    fn = getattr(getattr(ox, "distance", None), "nearest_nodes", None) or getattr(ox, "nearest_nodes", None)
    if fn is None:
        raise RuntimeError("当前 osmnx 版本未找到 nearest_nodes，请检查版本")
    return fn(G, X, Y)


def _admin_polygon():
    """从 admin_<adcode>.json 拼出城市行政边界多边形（shapely）；失败返回 None。
    用它取图比大 bbox 小很多、快很多（只取本市路网，不下载整个珠三角）。"""
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
        fp = DATA / f"admin_{CITY_ADMIN_ADCODE}.json"
        if not fp.exists():
            return None
        data = json.loads(fp.read_text(encoding="utf-8"))
        geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
        return unary_union(geoms) if geoms else None
    except Exception:
        return None


def _get_graph():
    global _GRAPH, _NODE_LL
    if _GRAPH is not None:
        return _GRAPH
    import osmnx as ox, time as _t, os as _os
    cache = DATA / f"_graph_{CITY}.graphml"
    G = None
    if cache.exists():
        print(f"[OSMnx] 载入缓存路网图 {cache.name} ...", flush=True)
        try:
            G = ox.load_graphml(cache)
        except Exception as e:                          # 缓存损坏(如上次写盘被中断) → 删除并重建
            print(f"[OSMnx] 缓存图损坏/不可读（{type(e).__name__}: {e}），删除并联网重建...", flush=True)
            try:
                cache.unlink()
            except Exception:
                pass
            G = None
    if G is None:
        poly = _admin_polygon(); t0 = _t.time()
        if poly is not None:
            print(f"[OSMnx] 按【{CITY_NAME}行政边界】联网取路网图（比大 bbox 小很多）...", flush=True)
            G = ox.graph_from_polygon(poly, network_type="drive")
        else:
            print(f"[OSMnx] 按 bbox 联网取路网图（bbox 很大时可能数分钟、占内存）...", flush=True)
            G = _ox_graph_from_bbox(ox, BBOX)
        G = _ox_add_traveltimes(ox, G)
        tmp = cache.with_name(cache.name + ".tmp")      # 原子写：先临时文件再改名，避免中断留坏缓存
        try:
            ox.save_graphml(G, tmp); _os.replace(str(tmp), str(cache))
            print(f"[OSMnx] 取图+缓存完成，用时 {_t.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[OSMnx] 缓存写入失败（{type(e).__name__}），本次用内存图继续。", flush=True)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    # 投影前先记下各节点的 WGS84 经纬度（x=lon, y=lat），供“需求映射到路网节点中心”用
    _NODE_LL = {int(n): (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
    _GRAPH = ox.project_graph(G, to_crs=f"EPSG:{CITY_CFG['metric_epsg']}")
    print(f"[OSMnx] 路网图就绪：{_GRAPH.number_of_nodes():,} 节点，{_GRAPH.number_of_edges():,} 边", flush=True)
    return _GRAPH


def _osmnx_ok():
    try:
        import osmnx  # noqa: F401
        return True
    except Exception:
        return False


def _events_to_nodes(lon, lat):
    """把低电量事件经纬度 snap 到最近路网节点，返回 node_id 数组（int64）。需 OSMnx 图。"""
    import osmnx as ox, pyproj
    _get_graph()
    tr = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{CITY_CFG['metric_epsg']}", always_xy=True)
    X, Y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(_ox_nearest_nodes(ox, _GRAPH, X, Y), dtype=np.int64)


def snap_to_nodes_ll(lon, lat):
    """把任意候选点坐标 snap 到最近**路网节点中心**，返回 (lon, lat)。
    仅在 network 后端且 OSMnx 可用时生效；haversine（无路网）则原样返回。
    用途：保证 S1 新增 / S3 置换的站点落在路网节点中心，而非 0.01° 网格中心。"""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    if DIST_BACKEND != "network" or not _osmnx_ok():
        return lon, lat
    try:
        ids = _events_to_nodes(lon, lat)
        nlon = np.array([_NODE_LL[int(n)][0] for n in ids])
        nlat = np.array([_NODE_LL[int(n)][1] for n in ids])
        return nlon, nlat
    except Exception:
        return lon, lat


def _snap(G, lon, lat):
    import osmnx as ox, pyproj
    tr = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{CITY_CFG['metric_epsg']}", always_xy=True)
    X, Y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(_ox_nearest_nodes(ox, G, X, Y))


def _networkx_distance_matrix(G, dem, sta, cutoff, big):
    import networkx as nx
    dem_rows = {}
    for i, n in enumerate(dem):
        dem_rows.setdefault(int(n), []).append(i)
    sta_cols = {}
    for j, n in enumerate(sta):
        sta_cols.setdefault(int(n), []).append(j)
    c = np.full((len(dem), len(sta)), big)
    Grev = G.reverse(copy=False)                         # 反向图：从站点跑最短路 = 原图 需求→站点
    for snode, jcols in tqdm(sta_cols.items(), desc=f"路网最短路·{len(sta_cols)}个站点节点(cutoff {cutoff/1000:.0f}km)"):
        try:
            lengths = nx.single_source_dijkstra_path_length(Grev, snode, cutoff=cutoff, weight="length")
        except Exception:
            continue
        for n, dm in lengths.items():
            rows = dem_rows.get(int(n))
            if rows:
                km = dm / 1000.0
                for j in jcols:
                    c[rows, j] = km
    return c


def _reverse_graph_csr(G):
    """把有向路网转成 scipy CSR 反向图；平行边取最短 length，避免 CSR 重复边求和。"""
    global _GRAPH_CSR_CACHE
    if _GRAPH_CSR_CACHE is not None and _GRAPH_CSR_CACHE[0] == id(G):
        return _GRAPH_CSR_CACHE[1:]
    from scipy.sparse import csr_matrix
    nodes = np.asarray(list(G.nodes()))
    node_to_idx = {int(n): i for i, n in enumerate(nodes)}
    best = {}
    for u, v, data in G.edges(data=True):
        w = float(data.get("length", np.inf))
        if not np.isfinite(w) or w < 0:
            continue
        # 反向边：在反向图从站点出发，等价于原图的 需求点→站点 距离。
        key = (node_to_idx[int(v)], node_to_idx[int(u)])
        old = best.get(key)
        if old is None or w < old:
            best[key] = w
    if not best:
        raise RuntimeError("路网图没有可用的 length 边，无法计算最短路")
    rc = np.asarray(list(best.keys()), dtype=np.int64)
    data = np.asarray(list(best.values()), dtype=float)
    csr = csr_matrix((data, (rc[:, 0], rc[:, 1])), shape=(len(nodes), len(nodes)))
    _GRAPH_CSR_CACHE = (id(G), nodes, node_to_idx, csr)
    return nodes, node_to_idx, csr


def _scipy_distance_matrix(G, dem, sta, cutoff, big):
    from scipy.sparse.csgraph import dijkstra
    _, node_to_idx, csr = _reverse_graph_csr(G)
    dem_idx = np.asarray([node_to_idx.get(int(n), -1) for n in dem], dtype=np.int64)
    valid_dem = dem_idx >= 0
    sta_cols = {}
    for j, n in enumerate(sta):
        if int(n) in node_to_idx:
            sta_cols.setdefault(int(n), []).append(j)
    sources = list(sta_cols)
    c = np.full((len(dem), len(sta)), big)
    if not sources or not valid_dem.any():
        return c
    batch = max(1, int(DIJKSTRA_BATCH))
    dem_pos = np.where(valid_dem)[0]
    dem_idx_valid = dem_idx[valid_dem]
    for start in tqdm(range(0, len(sources), batch),
                      desc=f"scipy最短路·{len(sources)}个站点节点(cutoff {cutoff/1000:.0f}km)"):
        src_nodes = sources[start:start + batch]
        src_idx = np.asarray([node_to_idx[int(n)] for n in src_nodes], dtype=np.int64)
        dist = dijkstra(csr, directed=True, indices=src_idx, limit=cutoff)
        dist = np.atleast_2d(dist)[:, dem_idx_valid] / 1000.0
        for r, snode in enumerate(src_nodes):
            finite = np.isfinite(dist[r])
            if not finite.any():
                continue
            rows = dem_pos[finite]
            vals = dist[r, finite]
            for j in sta_cols[snode]:
                c[rows, j] = vals
    return c


def network_matrix(blon, blat, slon, slat, cutoff_km=None, backend=None):
    """需求→站点路网最短路距离矩阵(km)。
    cutoff 默认等于 C_BAR：reach 判定和可达但很远的 access 成本都必须算到物理可达半径。
    scipy 后端把路网转成 CSR 稀疏图后跑 Dijkstra；networkx 后端保留作结果对拍。不可达填有限大数哨兵。"""
    G = _get_graph()
    cutoff_km = NET_CUTOFF_KM if cutoff_km is None else float(cutoff_km)
    if cutoff_km < C_BAR:
        print(f"[路网] 警告：cutoff {cutoff_km:g}km < C_BAR {C_BAR:g}km，会低估可达/出行成本。", flush=True)
    cutoff = cutoff_km * 1000.0
    print(f"[路网] 把 {len(blon)} 需求点 + {len(slon)} 站点 snap 到路网节点...", flush=True)
    dem = _snap(G, blon, blat); sta = _snap(G, slon, slat)
    big = C_BAR * 1e6                                     # 有限大数哨兵：>C_BAR 视为不可达，且 0×BIG=0 不产生 nan
    be = (backend or SHORTEST_PATH_BACKEND or "scipy").lower()
    if be == "networkx":
        return _networkx_distance_matrix(G, dem, sta, cutoff, big)
    try:
        return _scipy_distance_matrix(G, dem, sta, cutoff, big)
    except ImportError as e:
        print(f"[路网] scipy 不可用，回退 networkx 后端：{e}", flush=True)
        return _networkx_distance_matrix(G, dem, sta, cutoff, big)


def dist_matrix(blon, blat, slon, slat, backend=None):
    """统一入口：需求→站点 **OSMnx 路网最短路** 距离矩阵 (km)。只用路网最短路，无 haversine/网格近似。"""
    return network_matrix(blon, blat, slon, slat, backend=backend)


# ============================================================================
# 6. 2SFCA 错配指数（方法论 §4-§5，已数值验证）
# ============================================================================
def effective_capacity(fast, slow, gamma=GAMMA):
    """式(1) 有效容量 κ_j = γ·fast + slow（桩当量）。"""
    return gamma * np.asarray(fast, float) + np.asarray(slow, float)


def _disp_scale(w, fast, slow, u=SYS_UTIL, gamma=GAMMA):
    """式(1') 容量标尺 s=Σw/(u·Σκ)：把桩当量折成需求量 b=s·κ。增删站分析时固定在基线站集。"""
    tot = float(effective_capacity(fast, slow, gamma).sum())
    return float(np.sum(w)) / (u * tot) if tot > 0 else 0.0


def M_old(w, c, c_bar=None, p_dead=None):
    """旧指数（式10）：最近站硬指派、无限容量。零拥挤参照。返回 需求·km。"""
    cb = C_BAR if c_bar is None else c_bar; pd_ = P_DEAD if p_dead is None else p_dead
    mc = c.min(axis=1); cov = mc <= cb
    return float((w * mc * cov).sum() + pd_ * (w * (~cov)).sum())


def sfca_alloc(w, c, b, d0, c_bar=None):
    """式(2)(3)(4)(5) G2SFCA 引力(Huff)分配 —— 对数空间 softmax 数值稳定，无行为参数。
    可达性由物理半径判定（与 d0 解耦）。返回 p[i,j], reach_i, L_j负载, C_j=L_j/b_j 拥挤度(i2SFCA)。"""
    cb = C_BAR if c_bar is None else c_bar
    reach = (c <= cb).any(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logit = -0.5 * (c / d0) ** 2 + np.log(np.maximum(b, 1e-300))[None, :]
    logit = np.where(c <= cb, logit, -np.inf)
    mx = np.max(logit, axis=1, keepdims=True); mx = np.where(np.isfinite(mx), mx, 0.0)
    e = np.exp(logit - mx); ssum = e.sum(axis=1, keepdims=True)
    p = np.zeros_like(e); ok = ssum[:, 0] > 0; p[ok] = e[ok] / ssum[ok]
    L = (w[:, None] * p).sum(axis=0)
    Cj = np.where(b > 0, L / np.maximum(b, 1e-12), 0.0)
    return p, reach, L, Cj


def accessibility(w, c, b, d0, c_bar=None):
    """式(6) G2SFCA 需求侧可达性 A_i = Σ_j R_j G_ij, R_j=b_j/Σ_i' w_i' G_i'j。诊断字段。"""
    cb = C_BAR if c_bar is None else c_bar
    G = np.exp(-0.5 * (c / d0) ** 2); G[c > cb] = 0.0
    denom = (w[:, None] * G).sum(axis=0)
    R = np.where(denom > 0, b / np.maximum(denom, 1e-12), 0.0)
    return (R[None, :] * G).sum(axis=1)


def mismatch_M(w, c, fast, slow, d0=None, s=None, beta=None, d_ref=None,
               u=SYS_UTIL, gamma=GAMMA, c_bar=None, p_dead=None, parts=False):
    """式(7) 2SFCA 错配指数（需求·km）= M_access + M_crowd + M_reach。
    s=None 时按当前站集现算标尺；增删站分析须传入固定 s。
    d_ref=None 时用基线需求加权最近站距离作拥挤的 km 标尺（数据自定，非自由旋钮）。"""
    d0 = D0_DECAY if d0 is None else d0; beta = BETA_CROWD if beta is None else beta
    cb = C_BAR if c_bar is None else c_bar; pd_ = P_DEAD if p_dead is None else p_dead
    w = np.asarray(w, float); c = np.asarray(c, float)
    b = (s if s is not None else _disp_scale(w, fast, slow, u, gamma)) * effective_capacity(fast, slow, gamma)
    p, reach, L, Cj = sfca_alloc(w, c, b, d0, cb)
    M_access = float((w[:, None] * p * c).sum())
    sig = np.where(Cj > 1.0, 1.0 - 1.0 / np.maximum(Cj, 1e-12), 0.0)    # 式(8) 有界拥挤惩罚
    if d_ref is None:
        mc = np.where(c <= cb, c, np.inf).min(axis=1); rr = np.isfinite(mc)
        d_ref = float((w[rr] * mc[rr]).sum() / max(w[rr].sum(), 1e-12)) if rr.any() else 0.0
    M_crowd = float((w[:, None] * p * (beta * d_ref * sig)[None, :]).sum())
    M_reach = float(pd_ * w[~reach].sum())
    M = M_access + M_crowd + M_reach
    if parts:
        return dict(M=M, M_access=M_access, M_crowd=M_crowd, M_reach=M_reach,
                    L=L, C=Cj, reach=reach, d_ref=d_ref, b=b)
    return M


def baseline_report(w, c, fast, slow, s=None, d0=None):
    """基线打分：旧 M_old 参照 + 2SFCA 错配及其 access/crowd/reach 分解 + 拥挤度概览
    + **原生可达 KPI**（直接由物理可达半径 C_BAR 判定，不经 P_DEAD 折算，便于论文里单列）。
    旧字段全部保留；新增 reach_cov(可达需求量占比)、unreach_frac(不可达占比)、
    n_unreach(不可达需求点数)、w_unreach(不可达需求量)。"""
    r = mismatch_M(w, c, fast, slow, d0=d0, s=s, parts=True)
    w = np.asarray(w, float); reach = np.asarray(r["reach"], bool)
    w_tot = float(w.sum()); w_un = float(w[~reach].sum())
    reach_cov = float(w[reach].sum() / w_tot) if w_tot > 0 else 0.0   # 可达需求量占比（原生 KPI）
    return dict(M_old=M_old(w, c), M=r["M"], M_access=r["M_access"], M_crowd=r["M_crowd"],
                M_reach=r["M_reach"], C=r["C"], reach=r["reach"], d_ref=r["d_ref"],
                over_cap=int((r["C"] > 1).sum()), C_med=float(np.median(r["C"])),
                C_p90=float(np.percentile(r["C"], 90)),
                reach_cov=reach_cov, unreach_frac=float(1.0 - reach_cov),
                n_unreach=int((~reach).sum()), w_unreach=w_un)


# ============================================================================
# 7. 容量感知选站：S1 只增 / S2 只减 / S3 等量调配（贪心 + CELF，目标=2SFCA M）
# ============================================================================
def _candidate_pool(w, c, fast, slow, cand_dist, s, d0, beta, d_ref, nf, ns, pool_n, gamma, min_pool=0):
    """候选池：access 解析增益 top 一半 + crowd 缓解代理 top 一半（并集），兼顾“够不着/站满了”。"""
    r0 = mismatch_M(w, c, fast, slow, d0=d0, s=s, beta=beta, d_ref=d_ref, parts=True)
    if d_ref is None:
        d_ref = r0["d_ref"]
    b = r0["b"]; b_new = s * (gamma * nf + ns)
    G0 = np.exp(-0.5 * (c / d0) ** 2); G0[c > C_BAR] = 0.0
    KG = b[None, :] * G0; V = KG.sum(axis=1); Anum = (KG * c).sum(axis=1)
    base_acc = float((w * np.where(V > 0, Anum / np.maximum(V, 1e-12), 0.0)).sum())
    Gc = np.exp(-0.5 * (cand_dist / d0) ** 2); Gc[cand_dist > C_BAR] = 0.0
    g = b_new * Gc
    Vn = V[:, None] + g; numn = Anum[:, None] + g * cand_dist
    acc_new = (w[:, None] * np.where(Vn > 0, numn / np.maximum(Vn, 1e-12), 0.0)).sum(axis=0)
    gain_acc = base_acc - acc_new
    p0, _, _, _ = sfca_alloc(w, c, b, d0)
    sig0 = np.where(r0["C"] > 1.0, beta * d_ref * (1.0 - 1.0 / np.maximum(r0["C"], 1e-12)), 0.0)
    crowd_face = (p0 * sig0[None, :]).sum(axis=1)
    crowd_relief = ((w * crowd_face)[None, :] @ Gc).ravel()
    nC = cand_dist.shape[1]
    # 池子规模 = max(加速下限 pool_n, 策略目标 min_pool)，上限是候选总数 nC。
    # CAP_POOL/pool_n 只是加速旋钮，绝不把池子压到策略目标以下；唯一硬上限是候选地点数据量 nC。
    want = int(min(max(int(pool_n), int(min_pool)), nC))
    half = max(1, want // 2)
    pool = np.unique(np.concatenate([np.argsort(gain_acc)[::-1][:half],
                                     np.argsort(crowd_relief)[::-1][:half]]))
    if len(pool) < want:                       # 两路并集去重后不足，则按 access 增益补足到 want
        order = np.argsort(gain_acc)[::-1]
        extra = order[~np.isin(order, pool)]
        pool = np.concatenate([pool, extra[:want - len(pool)]])
    return pool, d_ref


def _celf_add_dense(w, c, cand_dist, pool, fast, slow, target, s, d0, beta, d_ref, nf, ns,
                    force_n, gamma, cb, pd_, desc):
    """稠密参照：每个 CELF 探测都 concat 一列、整体重算 mismatch_M。慢但直白，作对拍 oracle。
    同时记录每步的 M_access/M_crowd/M_reach 分解历史（提交后取 parts 一次）。"""
    cur_c, cur_f, cur_s = c, fast, slow
    r0 = mismatch_M(w, cur_c, cur_f, cur_s, d0=d0, s=s, beta=beta, d_ref=d_ref,
                    gamma=gamma, c_bar=cb, p_dead=pd_, parts=True)
    cur_M = r0["M"]; hist = [cur_M]
    hist_acc = [r0["M_access"]]; hist_crowd = [r0["M_crowd"]]; hist_reach = [r0["M_reach"]]
    avail = np.ones(len(pool), bool); gain = np.full(len(pool), np.inf); fresh = np.full(len(pool), -1)
    sel = []

    def _pad(n):
        hist.extend([cur_M] * n); hist_acc.extend([hist_acc[-1]] * n)
        hist_crowd.extend([hist_crowd[-1]] * n); hist_reach.extend([hist_reach[-1]] * n)

    for step in tqdm(range(target), desc=(desc or f"S1 加站[{CITY}]") + "·dense", leave=False):
        if not avail.any():
            _pad(target - step); break
        while True:                                                # CELF：只重算上界最高候选
            ids = np.where(avail)[0]; k = int(ids[np.argmax(gain[ids])])
            if fresh[k] == step:
                break
            caug = np.concatenate([cur_c, cand_dist[:, pool[k]:pool[k] + 1]], axis=1)
            fa = np.concatenate([cur_f, [nf]]); sa = np.concatenate([cur_s, [ns]])
            gain[k] = cur_M - mismatch_M(w, caug, fa, sa, d0=d0, s=s, beta=beta, d_ref=d_ref,
                                         gamma=gamma, c_bar=cb, p_dead=pd_); fresh[k] = step
        if (gain[k] <= 1e-9) and not force_n:
            _pad(target - step); break
        cur_c = np.concatenate([cur_c, cand_dist[:, pool[k]:pool[k] + 1]], axis=1)
        cur_f = np.concatenate([cur_f, [nf]]); cur_s = np.concatenate([cur_s, [ns]])
        rk = mismatch_M(w, cur_c, cur_f, cur_s, d0=d0, s=s, beta=beta, d_ref=d_ref,
                        gamma=gamma, c_bar=cb, p_dead=pd_, parts=True)
        cur_M = rk["M"]; avail[k] = False; sel.append(int(pool[k]))
        hist.append(cur_M); hist_acc.append(rk["M_access"]); hist_crowd.append(rk["M_crowd"]); hist_reach.append(rk["M_reach"])
    return dict(sel=sel, M=hist, M_access=hist_acc, M_crowd=hist_crowd, M_reach=hist_reach)


def _celf_add_fast(w, c, cand_dist, pool, fast, slow, target, s, d0, beta, d_ref, nf, ns,
                   force_n, gamma, cb, pd_, desc):
    """增量参照（默认）：与 _celf_add_dense 数值等价（合成数据对拍到机器精度、选站序列一致）。
    思路——加一个站只改“受影响行”的分配，故全程维护以下聚合量增量更新，避免每探测 concat 大矩阵+整重算：
      denom_i = Σ_j K_ij（K_ij=b_j·G_ij 引力核），inv_i=1/denom_i，Anum_i = Σ_j K_ij d_ij，
      L_j = Σ_i w_i p_ij 站负载，reach_i。并用恒等式 M_access=Σ_i w_i·Anum_i·inv_i、
      M_crowd=β·d_ref·Σ_j σ(C_j)·L_j（C_j=L_j/b_j），把每次评估降到一次 (w·Δinv)@K 矩乘。
    复杂度：每步 O(候选数 × nD×当前站数) 的矩乘，省掉了稠密版每探测的整套 exp+softmax+concat。"""
    w = np.asarray(w, float)
    nD, nF0 = c.shape
    b_t = float(s * (gamma * nf + ns))                  # 新站需求量标尺 b=s·κ（κ=γ·nf+ns）
    # 预分配引力核缓冲 K（nD×(nF0+target)），按列填充，避免每步 concat 复制大矩阵
    Kbuf = np.zeros((nD, nF0 + target)); b = np.zeros(nF0 + target)
    b[:nF0] = s * effective_capacity(fast, slow, gamma)
    G0 = np.exp(-0.5 * (c / d0) ** 2); G0[c > cb] = 0.0  # 高斯核，物理可达半径 cb 外清零
    Kbuf[:, :nF0] = b[:nF0][None, :] * G0
    del G0                                              # 及时释放临时核，省内存
    width = nF0
    denom = Kbuf[:, :width].sum(axis=1)
    reach = (c <= cb).any(axis=1)                       # reach 由 c≤cb 直接判（与核截断无关，band 行不冤判）
    inv = np.zeros_like(denom)
    np.divide(1.0, denom, out=inv, where=denom > 0)
    Anum = (Kbuf[:, :width] * c).sum(axis=1)
    L = (w * inv) @ Kbuf[:, :width]                     # 各站负载 L_j
    M_access = float((w * Anum * inv).sum())
    M_reach = float(pd_ * w[~reach].sum())

    def _crowd(Lv, bv):
        Cj = np.where(bv > 0, Lv / np.maximum(bv, 1e-12), 0.0)
        sig = np.where(Cj > 1.0, 1.0 - 1.0 / np.maximum(Cj, 1e-12), 0.0)
        return float(beta * d_ref * (sig * Lv).sum())

    crowd0 = _crowd(L, b[:width])
    cur_M = M_access + crowd0 + M_reach; hist = [cur_M]
    hist_acc = [M_access]; hist_crowd = [crowd0]; hist_reach = [M_reach]
    avail = np.ones(len(pool), bool); gain = np.full(len(pool), np.inf); fresh = np.full(len(pool), -1)
    cache = {}; sel = []

    def _eval(col):
        """评估“当前站集 + 候选列 col”的 M，返回 (M, L_aug, denom_t, inv_t, Anum_t, reach_t, kt)。"""
        gt = np.exp(-0.5 * (col / d0) ** 2); gt[col > cb] = 0.0
        kt = b_t * gt
        denom_t = denom + kt; inv_t = np.zeros_like(denom_t)
        np.divide(1.0, denom_t, out=inv_t, where=denom_t > 0)
        Anum_t = Anum + kt * col
        M_acc = float((w * Anum_t * inv_t).sum())
        reach_t = reach | (col <= cb); M_re = float(pd_ * w[~reach_t].sum())
        dinv = inv_t - inv                              # 仅受影响行非零
        L_exist = L + (w * dinv) @ Kbuf[:, :width]      # 现有站负载随分配重整而更新
        L_t = float((w * kt * inv_t).sum())             # 新站负载
        L_aug = np.concatenate([L_exist, [L_t]]); b_aug = np.concatenate([b[:width], [b_t]])
        cr = _crowd(L_aug, b_aug)
        return (M_acc + cr + M_re, L_aug, denom_t, inv_t, Anum_t, reach_t, kt, M_acc, cr, M_re)

    def _pad(n):                                        # 提前结束时用最后值补齐四条历史
        hist.extend([cur_M] * n); hist_acc.extend([hist_acc[-1]] * n)
        hist_crowd.extend([hist_crowd[-1]] * n); hist_reach.extend([hist_reach[-1]] * n)
    for step in tqdm(range(target), desc=desc or f"S1 加站[{CITY}]", leave=False):
        if not avail.any():
            _pad(target - step); break
        while True:                                     # CELF：只重算上界最高候选
            ids = np.where(avail)[0]; k = int(ids[np.argmax(gain[ids])])
            if fresh[k] == step:
                break
            res = _eval(cand_dist[:, pool[k]]); gain[k] = cur_M - res[0]; cache[k] = res; fresh[k] = step
        if (gain[k] <= 1e-9) and not force_n:
            _pad(target - step); break
        Mk, Lk, dk, ik, Ak, rk, ktk, acc_k, cr_k, re_k = cache[k]
        Kbuf[:, width] = ktk; b[width] = b_t            # 关键：新站容量标尺写入 b，否则拥挤项里 b=0 会丢站
        denom = dk; inv = ik; Anum = Ak; reach = rk; L = Lk; width += 1
        cur_M = Mk; avail[k] = False; sel.append(int(pool[k]))
        hist.append(cur_M); hist_acc.append(acc_k); hist_crowd.append(cr_k); hist_reach.append(re_k)
    return dict(sel=sel, M=hist, M_access=hist_acc, M_crowd=hist_crowd, M_reach=hist_reach)


def greedy_add(w, c, fast, slow, cand_lon, cand_lat, blon, blat, n_max, s=None,
               d0=None, beta=None, d_ref=None, new_fast=None, new_slow=None,
               pool_n=CAP_POOL, force_n=False, gamma=GAMMA, desc=None, backend=None):
    """S1：每步加一个让 2SFCA 错配 M 降得最多的候选站（建在需求点中心，按现有桩数中位赋容量）。
    返回 dict(sel=候选索引, M=每步后 M, new_fast, new_slow)。
    内层 CELF 评估按 backend 选 'fast'（增量，默认）或 'dense'（参照）；两者数值等价。"""
    d0 = D0_DECAY if d0 is None else d0; beta = BETA_CROWD if beta is None else beta
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    if s is None:
        s = _disp_scale(w, fast, slow)
    nf = float(max(1.0, np.median(fast))) if new_fast is None else new_fast
    ns = 0.0 if new_slow is None else new_slow
    cand_dist = dist_matrix(blon, blat, cand_lon, cand_lat)         # 需求×候选 距离（同后端）
    requested_n = int(max(0, n_max))
    # min_pool=requested_n：候选池至少要能容纳策略目标数（CAND_CAP/CAP_POOL 只是加速下限，不得压低规模）
    pool, d_ref = _candidate_pool(w, c, fast, slow, cand_dist, s, d0, beta, d_ref, nf, ns, pool_n, gamma,
                                  min_pool=requested_n)
    target = int(min(requested_n, len(pool)))
    if requested_n > target:
        print(f"[S1] 请求新增 {requested_n} 个，但可用候选地点(正需求路网节点) {int(cand_dist.shape[1])} 个、"
              f"成池 {target} 个；按 {target} 个执行。此为候选地点【数据上限】，非 CAND_CAP/CAP_POOL 限制"
              f"（二者已自动抬到不小于目标）。如需更大规模，请用更多抽样/天数提升正需求节点数。", flush=True)
    be = (backend or GREEDY_BACKEND or "fast").lower()
    loop = _celf_add_dense if be == "dense" else _celf_add_fast
    out = loop(w, c, cand_dist, pool, fast, slow, target, s, d0, beta, d_ref, nf, ns,
               force_n, gamma, C_BAR, P_DEAD, desc)
    out["new_fast"] = nf; out["new_slow"] = ns
    out["n_requested"] = requested_n; out["n_pool"] = int(len(pool)); out["n_executed_max"] = target
    return out


def greedy_remove(w, c, fast, slow, n_remove, s=None, d0=None, beta=None, d_ref=None, desc=None):
    """S2：每步删一个当前负载最低（最像容量冗余）的站。
    返回 dict(order, M, M_access, M_crowd, M_reach)（每步 mismatch_M(parts=True) 顺带记录分解）。"""
    d0 = D0_DECAY if d0 is None else d0; beta = BETA_CROWD if beta is None else beta
    fast = np.asarray(fast, float); slow = np.asarray(slow, float); nF = c.shape[1]
    if s is None:
        s = _disp_scale(w, fast, slow)
    r = mismatch_M(w, c, fast, slow, d0=d0, s=s, beta=beta, d_ref=d_ref, parts=True)
    if d_ref is None:
        d_ref = r["d_ref"]
    keep = np.ones(nF, bool); idx = np.arange(nF); order = []
    hist = [r["M"]]; hist_acc = [r["M_access"]]; hist_crowd = [r["M_crowd"]]; hist_reach = [r["M_reach"]]
    for _ in tqdm(range(int(min(max(n_remove, 0), nF - 1))), desc=desc or f"S2 删站[{CITY}]", leave=False):
        cols = idx[keep]
        if cols.size < 2:
            break
        j = int(np.argmin(r["L"]))
        keep[cols[j]] = False; order.append(int(cols[j]))
        r = mismatch_M(w, c[:, keep], fast[keep], slow[keep], d0=d0, s=s, beta=beta, d_ref=d_ref, parts=True)
        hist.append(r["M"]); hist_acc.append(r["M_access"]); hist_crowd.append(r["M_crowd"]); hist_reach.append(r["M_reach"])
    return dict(order=order, M=hist, M_access=hist_acc, M_crowd=hist_crowd, M_reach=hist_reach)


def swap(w, c, fast, slow, cand_lon, cand_lat, blon, blat, n_swap, s=None,
         d0=None, beta=None, d_ref=None, pool_n=CAP_POOL, gamma=GAMMA, desc=None,
         backend=None):
    """S3 等量调配：先贪心找最多 n_swap 个有价值新增站，再删同样数量最低负载站（净站数不变）。"""
    d0 = D0_DECAY if d0 is None else d0; beta = BETA_CROWD if beta is None else beta
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    if s is None:
        s = _disp_scale(w, fast, slow)
    if d_ref is None:
        d_ref = mismatch_M(w, c, fast, slow, d0=d0, s=s, beta=beta, parts=True)["d_ref"]
    target = int(max(0, min(n_swap, c.shape[1] - 1, len(cand_lon))))
    add = greedy_add(w, c, fast, slow, cand_lon, cand_lat, blon, blat, target, s=s, d0=d0, beta=beta,
                     d_ref=d_ref, pool_n=pool_n, force_n=True, desc=desc or f"S3 调配[{CITY}]",
                     backend=backend)
    rm = greedy_remove(w, c, fast, slow, target, s=s, d0=d0, beta=beta, d_ref=d_ref)["order"] if target > 0 else []
    added = add["sel"][:target]; rm = rm[:len(added)]
    nf, ns = add["new_fast"], add["new_slow"]
    cand_dist = dist_matrix(blon, blat, cand_lon, cand_lat)
    r0 = mismatch_M(w, c, fast, slow, d0=d0, s=s, beta=beta, d_ref=d_ref, parts=True)
    hist = [r0["M"]]; hist_acc = [r0["M_access"]]; hist_crowd = [r0["M_crowd"]]; hist_reach = [r0["M_reach"]]
    for k in range(1, len(added) + 1):
        keep = np.ones(c.shape[1], bool); keep[np.asarray(rm[:k], int)] = False
        c2 = np.concatenate([c[:, keep], cand_dist[:, np.asarray(added[:k], int)]], axis=1)
        f2 = np.concatenate([fast[keep], np.full(k, nf)]); s2 = np.concatenate([slow[keep], np.full(k, ns)])
        rk = mismatch_M(w, c2, f2, s2, d0=d0, s=s, beta=beta, d_ref=d_ref, parts=True)
        hist.append(rk["M"]); hist_acc.append(rk["M_access"]); hist_crowd.append(rk["M_crowd"]); hist_reach.append(rk["M_reach"])
    return dict(removed=rm, added=added, M=hist, M_access=hist_acc, M_crowd=hist_crowd,
                M_reach=hist_reach, new_fast=nf, new_slow=ns)


def counts_from_fracs(nF, fracs):
    """把“现有站数比例”集逐个转成整数站数（每个比例对应一个数，至少 1，**保持顺序、不去重**）。
    保持顺序/不去重是为了与各比例标签 zip 时一一对应、避免错位。S1/S2/S3 各传各的比例集
    （ADD_FRACS / REMOVE_FRACS / SWAP_FRACS）。"""
    from math import ceil
    return tuple(max(1, int(ceil(nF * f))) for f in fracs)


def add_counts(nF):
    """（兼容旧接口，行为与重构前一致）新增规模 = 现有站数 × ADD_FRACS，至少 1。
    等价于 counts_from_fracs(nF, ADD_FRACS)。"""
    return counts_from_fracs(nF, ADD_FRACS)


# ============================================================================
# 8/9/10 区块 + STEP_1~4 稳定接口（为保持单核心可读性集中追加于此）
#   notebook 只调用这些函数、不复制核心逻辑；每个函数标注其服务的 STEP 与逻辑区块。
# ============================================================================

# ---- 3b 数据审计辅助（STEP_1）----
def gps_quality_summary(days=None, smoke=False):
    """STEP_1：多日 GPS 质量汇总。返回每(车·日)点数数组、分段速度数组与统计。
    依赖 load_segments（优先走 .npz 缓存，无缓存才用 polars 读 parquet）。smoke=True 只取首日。"""
    days = (DAYS[:1] if smoke else DAYS) if days is None else days
    pts_list, v_list = [], []
    for ds, pth in days:
        z = load_segments(ds, pth)
        pts_list.append(np.bincount(z["vc"]))      # 每(车,日)点数
        v_list.append(z["v"][z["v"] > 0])
    pts = np.concatenate(pts_list) if pts_list else np.array([])
    v = np.concatenate(v_list) if v_list else np.array([])
    return dict(points_per_veh_day=pts, seg_speeds=v, n_days=len(days),
                n_veh_day=int(len(pts)), n_seg=int(len(v)),
                speed_median=float(np.median(v)) if v.size else float("nan"),
                min_track_points=MIN_TRACK_POINTS)


def sample_vehicle_trace(date_str=None, seed=None):
    """STEP_1：随机抽一辆点数达标的车，返回其当日完整轨迹与 SoC（用于‘看一辆真实车跑一天’）。
    返回 dict(veh, date, lon, lat, v, cum_km, soc)。"""
    if date_str is None:
        date_str = DAYS[0][0]
    z = load_segments(date_str)
    vc = z["vc"]; rng = np.random.default_rng(SEED if seed is None else seed)
    counts = np.bincount(vc); elig = np.where(counts >= MIN_TRACK_POINTS)[0]
    veh = int(rng.choice(elig)) if elig.size else int(np.argmax(counts))
    m = vc == veh
    cum = cumulative_kwh(z["d"], z["dt"], z["v"], z["start"])[m]
    ce = cum + (1.0 - SOC0["mean"]) * BATT
    soc = 1.0 - (ce - np.floor(ce / BATT) * BATT) / BATT
    return dict(veh=veh, date=date_str, lon=z["lon"][m], lat=z["lat"][m],
                v=z["v"][m], cum_km=np.cumsum(z["d"][m].astype(float)), soc=soc)


# ---- 4b 需求面辅助（STEP_2/STEP_3）----
_DEMAND_CACHE = {}   # 固定需求面进程内缓存：键=(CITY, draws, days签名, DEMAND_YEAR)

def simulate_soc_examples(percentiles=(30, 50, 70, 90), soc0=None):
    """STEP_2：代表车 SoC 曲线示例（按一天总耗电分位选车）。example_soc_curves 的语义封装。"""
    return example_soc_curves(percentiles=percentiles, soc0=soc0)


def build_demand_surface(draws=None, days=None, refresh=False):
    """STEP_2/STEP_3：构建并缓存‘固定低电量需求面’（路网节点中心，node_id 主键）。
    年度分析全程复用同一需求面（只变站点供给），故首次算好后进程内缓存，避免重复蒙特卡洛。
    draws/days 缺省取模块默认；refresh=True 强制重算。返回 demand_surface 的 dict。"""
    draws = N_ENSEMBLE if draws is None else int(draws)
    use_days = DAYS if days is None else days
    key = (CITY, int(draws), tuple(d for d, _ in use_days), DEMAND_YEAR)
    if refresh or key not in _DEMAND_CACHE:
        _DEMAND_CACHE[key] = demand_surface(draws=draws, days=use_days)
    return _DEMAND_CACHE[key]


# ---- 6b 布局打分 + 站点负载诊断（STEP_2/STEP_3）----
def score_layout(blon, blat, w, slon, slat, fast, slow, s, d_ref=None, d0=None):
    """给定一套站点布局，算 2SFCA 错配分解（固定容量标尺 s）。从 run_all 上移的稳定接口。
    内部需算需求→站点路网最短路距离矩阵。返回 dict(M, M_access, M_crowd, M_reach, reach_cov, over_cap, d_ref)。"""
    c = dist_matrix(blon, blat, slon, slat)
    r = mismatch_M(w, c, fast, slow, s=s, d_ref=d_ref, d0=d0, parts=True)
    w = np.asarray(w, float); reach = np.asarray(r["reach"], bool); wt = float(w.sum())
    return dict(M=r["M"], M_access=r["M_access"], M_crowd=r["M_crowd"], M_reach=r["M_reach"],
                reach_cov=float(w[reach].sum() / wt) if wt > 0 else 0.0,
                over_cap=int((r["C"] > 1).sum()), d_ref=r["d_ref"])


def station_load_report(w, c, fast, slow, s=None, d0=None, sid=None, slon=None, slat=None):
    """STEP_2：每站可达性/负载/拥挤诊断（纯 numpy，给定距离矩阵 c）。
    返回站级 L_j(负载)、C_j(拥挤度=L/b)、b_j(容量标尺)、is_over(C>1)，及需求侧 A_i(可达性)。
    用于站点负载图、拥挤度图、负载密度分布。"""
    d0 = D0_DECAY if d0 is None else d0
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    if s is None:
        s = _disp_scale(w, fast, slow)
    r = mismatch_M(w, c, fast, slow, d0=d0, s=s, parts=True)
    A = accessibility(w, c, r["b"], d0)
    return dict(L=r["L"], C=r["C"], b=r["b"], is_over=(r["C"] > 1.0), A_demand=A, d_ref=r["d_ref"],
                sid=None if sid is None else np.asarray(sid),
                slon=None if slon is None else np.asarray(slon, float),
                slat=None if slat is None else np.asarray(slat, float))


# ---- 7b 三类策略薄封装（STEP_3）：各用各的比例 ----
def run_s1_add(w, c, fast, slow, cand_lon, cand_lat, blon, blat, fracs=None,
               s=None, d_ref=None, pool_n=CAP_POOL, force_n=True, backend=None):
    """S1 只增：按 ADD_FRACS（或传入 fracs）定最大加站数，贪心+CELF 加站。返回 greedy_add 结果 + fracs/counts。"""
    fracs = ADD_FRACS if fracs is None else fracs
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    nF = c.shape[1]; counts = counts_from_fracs(nF, fracs); n_max = int(max(counts))
    if s is None:
        s = _disp_scale(w, fast, slow)
    out = greedy_add(w, c, fast, slow, cand_lon, cand_lat, blon, blat, n_max,
                     s=s, d_ref=d_ref, pool_n=max(int(pool_n), n_max),
                     force_n=force_n, desc="S1 只增", backend=backend)
    actual_counts = tuple(min(int(v), int(out.get("n_executed_max", n_max))) for v in counts)
    out.update(fracs=tuple(fracs), counts=counts, actual_counts=actual_counts,
               n_max=n_max, n_executed_max=int(out.get("n_executed_max", n_max)))
    return out


def run_s2_remove(w, c, fast, slow, fracs=None, s=None, d_ref=None):
    """S2 只减：按 REMOVE_FRACS 定最大减站数，逐步删最低负载站。返回 greedy_remove 结果 + fracs/counts。"""
    fracs = REMOVE_FRACS if fracs is None else fracs
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    nF = c.shape[1]; counts = counts_from_fracs(nF, fracs); n_max = int(min(max(counts), nF - 1))
    if s is None:
        s = _disp_scale(w, fast, slow)
    out = greedy_remove(w, c, fast, slow, n_max, s=s, d_ref=d_ref, desc="S2 只减")
    out.update(fracs=tuple(fracs), counts=counts, n_max=n_max)
    return out


def run_s3_swap(w, c, fast, slow, cand_lon, cand_lat, blon, blat, fracs=None,
                s=None, d_ref=None, pool_n=CAP_POOL, backend=None):
    """S3 等量调配：按 SWAP_FRACS 定调配规模，加同量、删同量（净站数不变）。返回 swap 结果 + fracs/counts。"""
    fracs = SWAP_FRACS if fracs is None else fracs
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    nF = c.shape[1]; counts = counts_from_fracs(nF, fracs); n_max = int(min(max(counts), nF - 1))
    if s is None:
        s = _disp_scale(w, fast, slow)
    out = swap(w, c, fast, slow, cand_lon, cand_lat, blon, blat, n_max,
               s=s, d_ref=d_ref, pool_n=max(int(pool_n), n_max), backend=backend)
    out.update(fracs=tuple(fracs), counts=counts, n_max=n_max)
    return out


# ---- 8 年度循环 + 真实新增对照（STEP_3）----
def _candidates_from_demand(D, cap=None, smoke=False):
    """候选站集 = 需求最高的若干路网节点（snap 到节点中心）。返回 (clon, clat)。"""
    cap = (80 if smoke else CAND_CAP) if cap is None else int(cap)
    w = D["w"]; lc = D["lon_c"]; ac = D["lat_c"]
    cand = np.argsort(w)[::-1][:min(cap, int((w > 0).sum()))]
    return snap_to_nodes_ll(lc[cand], ac[cand])


def run_yearly_scenarios(years=None, demand=None, draws=None, run_scenarios=True,
                         smoke=False, verbose=True):
    """STEP_3 主循环：固定需求面，对每个年份用 station_snapshot(Y) 做 baseline + S1/S2/S3，
    并在存在 Y+1 时附带 real_additions_between(Y,Y+1) 供对照。
    返回 dict[year]={stations, baseline, s_fix, d_ref, [s1,s2,s3], real_next}。
    **防泄漏：每年只用截至 Y 的存量；real_next 仅作附带字段返回，不进入候选/打分。**"""
    years = station_years() if years is None else list(years)
    D = build_demand_surface(draws=draws) if demand is None else demand
    w = D["w"]; lc = D["lon_c"]; ac = D["lat_c"]
    max_strategy_n = 0
    if run_scenarios:
        max_nF = max(len(station_snapshot(Y)) for Y in years) if years else 0
        max_strategy_n = max(max(counts_from_fracs(max_nF, ADD_FRACS)),
                             max(counts_from_fracs(max_nF, SWAP_FRACS))) if max_nF else 0
    cand_cap = max((80 if smoke else CAND_CAP), max_strategy_n)
    clon, clat = _candidates_from_demand(D, cap=cand_cap, smoke=smoke)
    pool_n = max(20 if smoke else CAP_POOL, max_strategy_n)
    out = {}
    for Y in years:
        st = station_snapshot(Y)
        if len(st) < 2:
            if verbose:
                print(f"[{Y}] 存量 {len(st)} 站，过少，跳过")
            continue
        slon = st["lon"].to_numpy(); slat = st["lat"].to_numpy()
        fast = st["fast"].to_numpy(); slow = st["slow"].to_numpy()
        c = dist_matrix(lc, ac, slon, slat)
        s_fix = _disp_scale(w, fast, slow)
        base = baseline_report(w, c, fast, slow, s=s_fix); d_ref = base["d_ref"]
        rec = dict(stations=st, baseline=base, s_fix=s_fix, d_ref=d_ref)
        if run_scenarios:
            rec["s1"] = run_s1_add(w, c, fast, slow, clon, clat, lc, ac, s=s_fix, d_ref=d_ref, pool_n=pool_n)
            rec["s2"] = run_s2_remove(w, c, fast, slow, s=s_fix, d_ref=d_ref)
            rec["s3"] = run_s3_swap(w, c, fast, slow, clon, clat, lc, ac, s=s_fix, d_ref=d_ref, pool_n=pool_n)
        nxt = Y + 1
        rec["real_next"] = real_additions_between(Y, nxt) if nxt <= max(years) else None  # 只读，不回流
        out[Y] = rec
        if verbose:
            print(f"[{Y}] 存量 {len(st)} | M={base['M']:.0f} "
                  f"(acc{base['M_access']:.0f}+crowd{base['M_crowd']:.0f}+reach{base['M_reach']:.0f}) "
                  f"| 可达覆盖 {base['reach_cov']:.3f}")
    return out


def compare_recommendations_to_real(rec_lon, rec_lat, real_lon, real_lat,
                                    thresholds=None, admin_of=None):
    """STEP_3：模型推荐新增点 vs 下一年真实新增点的对照指标（真实新增只读）。
    - nearest_km：每个真实新增到最近推荐点的球面距离(km)。
    - hit_rate：真实新增落在任一推荐点 <= 阈值 的比例（按 MATCH_THRESH_KM）。
    admin_of：可选 callable(lon,lat)->区名，给行政区计数重合（缺省不算）。
    返回 dict(nearest_km, hit_rate{thr:rate}, n_real, n_rec, median_nearest[, admin_real/rec])。"""
    thresholds = MATCH_THRESH_KM if thresholds is None else thresholds
    rec_lon = np.asarray(rec_lon, float); rec_lat = np.asarray(rec_lat, float)
    real_lon = np.asarray(real_lon, float); real_lat = np.asarray(real_lat, float)
    if rec_lon.size == 0 or real_lon.size == 0:
        return dict(nearest_km=np.array([]), hit_rate={float(t): float("nan") for t in thresholds},
                    n_real=int(real_lon.size), n_rec=int(rec_lon.size), median_nearest=float("nan"))
    nearest = np.array([float(np.min(haversine_km(x, y, rec_lon, rec_lat)))
                        for x, y in zip(real_lon, real_lat)])
    res = dict(nearest_km=nearest, hit_rate={float(t): float((nearest <= t).mean()) for t in thresholds},
               n_real=int(real_lon.size), n_rec=int(rec_lon.size), median_nearest=float(np.median(nearest)))
    if admin_of is not None:
        try:
            res["admin_real"] = pd.Series([admin_of(x, y) for x, y in zip(real_lon, real_lat)]).value_counts().to_dict()
            res["admin_rec"] = pd.Series([admin_of(x, y) for x, y in zip(rec_lon, rec_lat)]).value_counts().to_dict()
        except Exception:
            pass
    return res


# ---- 8 参数敏感性（STEP_4）----
def run_sensitivity(w, c, fast, slow, grid=None, s_mode="refit"):
    """STEP_4：单参数敏感性（纯 numpy，给定距离矩阵 c）。grid 缺省取 SENS_GRID。
    每次只动一个参数、其余固定；s_mode='refit' 时对改变容量标尺的 u/gamma 重算 s，其余用基线 s。
    返回 tidy DataFrame：param,value,M,M_access,M_crowd,M_reach,reach_cov,over_cap。
    ★ P_DEAD 行尤其看 reach_cov：它是原生可达覆盖率，不随 P_DEAD 改变；M_reach 会变。"""
    grid = SENS_GRID if grid is None else grid
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    w = np.asarray(w, float); wt = float(w.sum())
    s_base = _disp_scale(w, fast, slow)
    rows = []
    for p, vals in grid.items():
        for v in vals:
            kw = dict(d0=D0_DECAY, beta=BETA_CROWD, gamma=GAMMA, c_bar=C_BAR, p_dead=P_DEAD); u = SYS_UTIL
            if p == "d0": kw["d0"] = v
            elif p == "beta": kw["beta"] = v
            elif p == "gamma": kw["gamma"] = v
            elif p == "C_BAR": kw["c_bar"] = v
            elif p == "P_DEAD": kw["p_dead"] = v
            elif p == "u": u = v
            else: raise KeyError(f"未知敏感性参数 {p}")
            s = (_disp_scale(w, fast, slow, u=u, gamma=kw["gamma"])
                 if (s_mode == "refit" and p in ("u", "gamma")) else s_base)
            r = mismatch_M(w, c, fast, slow, d0=kw["d0"], s=s, beta=kw["beta"], gamma=kw["gamma"],
                           c_bar=kw["c_bar"], p_dead=kw["p_dead"], parts=True)
            reach = np.asarray(r["reach"], bool)
            rows.append(dict(param=p, value=v, M=r["M"], M_access=r["M_access"],
                             M_crowd=r["M_crowd"], M_reach=r["M_reach"],
                             reach_cov=float(w[reach].sum() / wt) if wt > 0 else 0.0,
                             over_cap=int((r["C"] > 1).sum())))
    return pd.DataFrame(rows)


# ---- 9 可视化（薄封装，调用 style 原语；研究语义图。画到传入 ax，由 notebook 控制 fig/savefig）----
def plot_demand_network(ax, D, slon=None, slat=None, norm=None, cmap=None, title=None):
    """STEP_2/3：需求路网着色图（不画散点）。用 node_id 关联路段、max(w_u,w_v) 着色；可叠现有站 marker。
    返回 colorbar 用的 LineCollection（无路网缓存时回退行政边界并返回 None）。"""
    import matplotlib.colors as mcolors, style
    w = D["w"]; m = w > 0
    if norm is None:
        vmax = float(np.percentile(w[m], 98)) if np.any(m) else 1.0
        norm = mcolors.PowerNorm(gamma=0.72, vmin=0.0, vmax=max(vmax, 1e-9))
    lc = style.draw_network_demand(ax, D.get("node_id"), w, cache_dir=DATA, city=CITY, norm=norm, cmap=cmap)
    if lc is None:
        try:
            style.draw_admin(ax, cache_dir=DATA, adcode=CITY_ADMIN_ADCODE, color="#7E8795", lw=0.7)
        except Exception:
            pass
    if slon is not None:
        ax.scatter(slon, slat, s=2.5, c="#5B6573", marker=".", alpha=0.4, zorder=4, label="现有站")
    if title:
        ax.set_title(title)
    ax.set_xlabel("经度"); ax.set_ylabel("纬度"); ax.set_aspect("equal", adjustable="box")
    return lc


def plot_scenario_curve(ax, M, counts=None, labels=None, color=None, label="2SFCA 错配 M"):
    """STEP_3：情景相对降幅曲线 M/M_base vs 变化站点数；在各比例节点打点标注。"""
    import style
    M = np.asarray(M, float); base = M[0] if M.size and M[0] else 1.0
    y = M / base; x = np.arange(len(M)); color = color or style.C["blue"]
    ax.plot(x, y, "-", lw=2.0, color=color, label=label)
    if counts is not None:
        xs = [int(n) for n in counts if int(n) < len(M)]
        ax.scatter(xs, [y[i] for i in xs], s=26, color=color, zorder=5)
        if labels is not None:
            for lab, xx in zip(labels, xs):
                ax.annotate(lab, (xx, y[xx]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=color)
    ax.axhline(1.0, color=style.C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("变化站点数 N"); ax.set_ylabel("相对错配指数 M / M_base"); ax.legend(loc="best")


def plot_scenario_decomp(ax, r, title=None, counts=None, fracs=None, sign=""):
    """STEP_3/run_all：情景降幅曲线——总 M/M_base **加** 三项分解各自相对基线的比值
    （M_access/M_access_base、M_crowd/M_crowd_base、M_reach/M_reach_base）。
    并在各**比例节点（站点变化 %）**于【总 M 曲线】上打点标注：比例 % 与 总 M 降幅 ↓%。
    counts/fracs 缺省取 r 的 actual_counts/counts 与 fracs；sign 用 '+'/'-'/'±' 区分 S1/S2/S3。
    某分量基线为 0（如全可达时 M_reach_base=0）则跳过该比值线，避免除零。"""
    import style
    M = np.asarray(r["M"], float); x = np.arange(len(M))
    base = M[0] if M.size and M[0] else 1.0
    yM = M / base
    series = [("M", style.C["ink"], "M (总)", "-", 2.0),
              ("M_access", style.C["blue"], "M_access", "--", 1.2),
              ("M_crowd", style.C["orange"], "M_crowd", "--", 1.2),
              ("M_reach", style.C["red"], "M_reach", "--", 1.2)]
    for key, col, lab, ls, lw in series:
        if key not in r:
            continue
        arr = np.asarray(r[key], float)
        if arr.size != M.size or not np.isfinite(arr[0]) or arr[0] <= 0:
            continue
        ax.plot(x, arr / arr[0], ls, lw=lw, color=col, label=f"{lab}/base")
    # 比例节点（站点变化 %）：在总 M 曲线上打点 + 标注 比例% 与 总 M 降幅↓%
    if counts is None:
        counts = r.get("actual_counts", r.get("counts"))
    if fracs is None:
        fracs = r.get("fracs")
    if counts is not None and fracs is not None:
        for n, f in zip(np.asarray(counts, int), np.asarray(fracs, float)):
            n = int(n)
            if not (0 < n < len(M)):
                continue
            yy = float(yM[n]); drop = max(0.0, (1.0 - yy) * 100.0)
            ax.axvline(n, color=style.C["light"], lw=0.6, zorder=0)
            ax.scatter([n], [yy], s=24, color=style.C["ink"], edgecolor="white", linewidth=0.5, zorder=6)
            ax.annotate(f"{sign}{f * 100:.0f}%\n↓{drop:.1f}%", (n, yy),
                        textcoords="offset points", xytext=(0, 7), ha="center", va="bottom",
                        fontsize=6.2, color=style.C["ink"])
    ax.axhline(1.0, color=style.C["gray"], ls=":", lw=0.8)
    ax.set_xlabel("变化站点数 N"); ax.set_ylabel("相对基线比值 (·/·_base)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=6)
    ax.margins(x=0.03, y=0.12)


def plot_change_map(ax, D, slon, slat, add_lon=None, add_lat=None, remove_idx=None,
                    real_lon=None, real_lat=None, norm=None, title=None):
    """STEP_3：变化地图——需求路段着色底图 + 现有站(灰点) + 推荐新增(绿星) + 关闭(红叉) + 真实新增(蓝三角)。
    现有/新增/真实都是站点 marker（不是需求点），故用点/星/叉/三角。返回 colorbar 用的 LineCollection 或 None。"""
    lc = plot_demand_network(ax, D, norm=norm)
    slon = np.asarray(slon, float); slat = np.asarray(slat, float)
    import style
    ax.scatter(slon, slat, s=2.2, c=style.C["gray"], marker=".", alpha=0.35, zorder=4, label="现有站")
    if remove_idx is not None and np.asarray(remove_idx).size:
        ri = np.asarray(remove_idx, int)
        ax.scatter(slon[ri], slat[ri], s=28, c=style.C["red"], marker="x", linewidths=1.0, zorder=5, label="关闭")
    if real_lon is not None and np.asarray(real_lon).size:
        ax.scatter(np.asarray(real_lon, float), np.asarray(real_lat, float), s=34, facecolors="none",
                   edgecolors=style.C["purple"], marker="^", linewidths=0.9, zorder=6, label="真实新增(下一年)")
    if add_lon is not None and np.asarray(add_lon).size:
        ax.scatter(np.asarray(add_lon, float), np.asarray(add_lat, float), s=92, c=style.C["green"],
                   marker="*", edgecolor="white", linewidths=0.6, zorder=7, label="推荐新增(路网节点)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", markerscale=1.2)
    return lc


# ---- 10 冒烟测试（端到端；需 osmnx + 路网缓存，适合在本地 Anaconda 跑）----
def _smoke_pipeline(year=None, draws=4, verbose=True):
    """最小端到端冒烟：单日少抽样需求面 → 某年存量 → 距离矩阵 → M 分解 → S1/S2/S3 → 真实新增对照。
    仅用于‘跑通链路’，不写正式 Outputs。返回各环节关键量的 dict。"""
    days = DAYS[:1]
    D = build_demand_surface(draws=draws, days=days, refresh=True)
    yrs = station_years(); Y = yrs[len(yrs) // 2] if year is None else int(year)
    st = station_snapshot(Y)
    slon = st["lon"].to_numpy(); slat = st["lat"].to_numpy()
    fast = st["fast"].to_numpy(); slow = st["slow"].to_numpy()
    w = D["w"]; lc = D["lon_c"]; ac = D["lat_c"]
    c = dist_matrix(lc, ac, slon, slat); s_fix = _disp_scale(w, fast, slow)
    base = baseline_report(w, c, fast, slow, s=s_fix)
    clon, clat = _candidates_from_demand(D, smoke=True)
    r1 = run_s1_add(w, c, fast, slow, clon, clat, lc, ac, fracs=(0.05,), s=s_fix, d_ref=base["d_ref"], pool_n=20)
    r2 = run_s2_remove(w, c, fast, slow, fracs=(0.10,), s=s_fix, d_ref=base["d_ref"])
    out = dict(year=Y, n_demand=D["n_cells"], n_station=len(st), M=base["M"],
               reach_cov=base["reach_cov"], s1_end=r1["M"][-1], s2_end=r2["M"][-1])
    if verbose:
        print("[_smoke_pipeline]", out)
    return out


# 模块导入时配置默认城市，使常量（BBOX 等）可用
configure_city(CITY)
