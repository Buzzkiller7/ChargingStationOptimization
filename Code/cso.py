# -*- coding: utf-8 -*-
"""
cso.py — 充电站错配指数 M(F) 的单文件计算引擎（干净重构版）。

整条链路：
  轨迹 GPS → 逐段能耗 → 电量 SoC → 低电量需求面 w_i
  → 需求格到站点距离 c_ij
  → 三个错配指数：M_old(无限容量，式1) / M_disp(空间置换 LP，式3-4) / M_queue(时间排队 M/M/c，式5-7)
  → S1 只增 / S2 只减 / S3 等量调配（直接在容量口径上选站）

约定：时间一律用“小时”；距离用 km；M_disp 单位需求·km；M_queue 单位需求·min。
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

try:                                                   # 进度条；未装 tqdm 则退化为普通 range
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **k): return it

# ============================================================================
# 0. 参数（方法论 §8）—— 全部集中在这里，便于审阅与扫动
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent          # 项目根（Code 的上一级）
BASE_DATA = ROOT / "data"                               # 原始数据根目录；多城市时优先读 data/<city>/

# 城市配置只描述“数据在哪里、边界在哪里、站点表字段叫什么”。广州是默认配置；
# 后续跑全国城市时，可以在外部 JSON 里加入同样结构的城市配置，无需改代码。
CITY_CONFIGS = {
    "guangzhou": {
        "name_cn": "广州",
        "trace_file": "Taxi_2019_10_14_admin_4401.parquet",
        "station_file": "guangzhou_station.csv",
        "trace_date": "2019-10-14",
        "bbox": dict(lon_min=112.90, lon_max=114.10, lat_min=22.50, lat_max=24.00),
        "admin_adcode": "440100",
        "station_cols": dict(lon="WGS84_station_lg", lat="WGS84_station_lt",
                             fast="station_fast_cnt", slow="station_slow_cnt",
                             create_time="create_time", sid="station_id"),
    }
}

CITY = os.environ.get("CSO_CITY", "guangzhou").strip().lower() or "guangzhou"
CITY_NAME = "广州"
CITY_ADMIN_ADCODE = "440100"
DATA = BASE_DATA
TRACE_FILE = ""
STATION_FILE = ""
TRACE_DATE = ""
BBOX = {}
STATION_COLS = {}


def _load_city_configs(config_path=None):
    """读取城市配置。外部 JSON 可以是 {city: config}，也可以是 {"cities": {city: config}}。"""
    configs = dict(CITY_CONFIGS)
    path = config_path or os.environ.get("CSO_CITY_CONFIG", "").strip()
    if path:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            extra = json.load(f)
        if "cities" in extra:
            extra = extra["cities"]
        for k, v in extra.items():
            configs[str(k).lower()] = v
    return configs


def configure_city(city=None, config_path=None):
    """切换当前城市配置。

    城市配置会影响轨迹文件、站点文件、bbox、行政区边界 adcode 和输出目录标签。
    bbox 是经纬度裁剪框，用来限定轨迹与站点的空间范围；全国多城市运行时必须逐城设置，
    否则会把别的城市数据错误裁掉或错误纳入。
    """
    global CITY, CITY_NAME, CITY_ADMIN_ADCODE, DATA, TRACE_FILE, STATION_FILE, TRACE_DATE, BBOX, STATION_COLS
    configs = _load_city_configs(config_path)
    key = (city or os.environ.get("CSO_CITY", CITY) or "guangzhou").strip().lower()
    if key not in configs:
        known = ", ".join(sorted(configs))
        raise KeyError(f"未知城市配置：{key}。请在 --city-config JSON 中补充；当前可用：{known}")
    cfg = configs[key]
    CITY = key
    CITY_NAME = str(cfg.get("name_cn") or cfg.get("name") or key)
    CITY_ADMIN_ADCODE = str(cfg.get("admin_adcode", "") or "")
    TRACE_FILE = str(cfg["trace_file"])
    STATION_FILE = str(cfg["station_file"])
    TRACE_DATE = str(cfg["trace_date"])
    BBOX = dict(cfg["bbox"])
    STATION_COLS = dict(cfg["station_cols"])
    # 多城市推荐 data/<city>/；为兼容当前广州项目，如果子目录不存在就回退到 data/。
    city_data = BASE_DATA / CITY
    DATA = city_data if city_data.exists() else BASE_DATA
    return cfg


configure_city(CITY)

# 空间栅格
GRID_DEG = 0.01                  # 0.01° ≈ 1 km 的需求格
ENC = 100000                     # block_id = row*ENC + col 的编码基

# 车型能耗（时间用小时）：dE = d·(k_d + k_v2·v²) + dt·k_t
BATT = 78.4                      # 电池容量 kWh（Tesla Model 3 LR）
K_D = 0.150                      # kWh/km 行驶里程能耗
K_V2 = 0.000025                  # kWh·h²/km³ 速度平方项（空气阻力）
K_T = 0.9                        # kWh/h 怠速/附属功率（≈0.9 kW）

# 需求定义
SOC_LOW = 0.20                   # 低电量触发阈值
SEG_DT_CLIP = 0.5                # 截断异常时间间隔 [h]
SEG_D_CLIP = 50.0                # 截断异常距离跳点 [km]
MIN_TRACK_POINTS = 100           # 出行点数<100 的车剔除
SOC0 = dict(mean=0.85, sd=0.10, lo=0.40, hi=1.00)       # 初始 SoC 截断正态

# 成本 / 可达
DETOUR = 1.3                     # 直线→路网绕行系数
C_BAR = 100.0                    # 低电量车可达里程 [km]（≈SOC_LOW·BATT/K_D）
P_DEAD = 1000.0                  # 抛锚惩罚（与 w·c 同量纲，等效 km）

# 容量双口径（方法论 §4、§5）
GAMMA = 10.0                     # 快/慢桩有效容量比 κ=γ·fast+slow（=μ_fast/μ_slow）
SYS_UTIL = 1.0                   # 系统利用率 u（总需求/总容量）；空间置换用，扫 0.8–1.5
DISP_LP_KNN = 30                 # 空间置换 LP：每格只连最近 30 个可达站（压规模）
MU_FAST = 2.0                    # 快充服务率 辆/h（25–40min 充好）
MU_SLOW = 0.2                    # 慢充服务率 辆/h（数小时）
OPER_HOURS = 14.0                # 有效营业时长 H [h]，把日需求折成时均到达率 λ=L/H
WAIT_BALK_MIN = 30.0             # 司机愿排队上限（分钟），平均等待按此封顶
AVG_SPEED = 30.0                 # 代表车速 km/h（距离→分钟）

# 集成 / 选站
N_ENSEMBLE = 40                  # 初始 SoC 蒙特卡洛抽样数（正式精度；冒烟可调小）
ADD_FRACS = (0.01, 0.02, 0.05, 0.1)   # S1/S2/S3 规模 = 现有站数的 1/2/5/10%
CAND_CAP = 1200                  # 候选需求格上限
CAP_POOL = 60                    # 容量感知选站的候选站池上限（惰性贪心 CELF 只首轮全评、之后每步仅重算少数）
SEED = 25


# ============================================================================
# 1. 几何 / 栅格
# ============================================================================
def haversine_km(lon1, lat1, lon2, lat2):
    """经纬度球面距离 (km)，支持 numpy 广播。"""
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(d))


def to_block(lon, lat):
    """经纬度 → bbox 内规则栅格的整数 block_id。"""
    ii = np.clip(((lat - BBOX["lat_min"]) / GRID_DEG).astype(np.int64), 0, ENC - 1)
    jj = np.clip(((lon - BBOX["lon_min"]) / GRID_DEG).astype(np.int64), 0, ENC - 1)
    return ii * ENC + jj


def block_centroid(block_id):
    """block_id → 栅格中心经纬度。"""
    ii = block_id // ENC; jj = block_id % ENC
    return (BBOX["lon_min"] + (jj + 0.5) * GRID_DEG), (BBOX["lat_min"] + (ii + 0.5) * GRID_DEG)


# ============================================================================
# 2. 数据加载
# ============================================================================
def load_stations(mode="truncated", return_stats=False):
    """读站点表。保留 bbox 内、坐标有效、容量>0 的站。
    mode='truncated'：只保留轨迹观测日(含)之前已建成的站（因果诚实）。
    mode='comprehensive'：不按建成日过滤，纳入全部现存站（反事实评估）。
    返回 DataFrame(lon,lat,fast,slow,sid)。"""
    sc = STATION_COLS
    raw = pd.read_csv(DATA / STATION_FILE)
    lon = pd.to_numeric(raw[sc["lon"]], errors="coerce")
    lat = pd.to_numeric(raw[sc["lat"]], errors="coerce")
    fast = pd.to_numeric(raw[sc["fast"]], errors="coerce").fillna(0)
    slow = pd.to_numeric(raw[sc["slow"]], errors="coerce").fillna(0)
    ok = (lon.notna() & lat.notna()
          & lon.between(BBOX["lon_min"], BBOX["lon_max"])
          & lat.between(BBOX["lat_min"], BBOX["lat_max"])
          & ((fast + slow) > 0))
    n_future = 0
    if mode == "truncated":
        cutoff = pd.Timestamp(TRACE_DATE)
        created = pd.to_datetime(raw[sc["create_time"]], errors="coerce")
        date_ok = created.notna() & (created.dt.normalize() <= cutoff)
        n_future = int((created.dt.normalize() > cutoff).sum())
        ok = ok & date_ok
    st = pd.DataFrame(dict(lon=lon[ok].to_numpy(float), lat=lat[ok].to_numpy(float),
                           fast=fast[ok].to_numpy(float), slow=slow[ok].to_numpy(float),
                           sid=raw.loc[ok, sc["sid"]].astype(str).to_numpy())).reset_index(drop=True)
    stats = dict(mode=mode, raw=int(len(raw)), kept=int(len(st)),
                 dropped=int(len(raw) - len(st)), future_create=n_future,
                 zero_cap=int(((fast + slow) <= 0).sum()))
    return (st, stats) if return_stats else st


def load_segments():
    """读轨迹 parquet，裁剪 bbox、剔短轨迹、算相邻点的距离 d(km)/时间 dt(h)/速度 v(km/h)。
    返回 dict(vc, lon, lat, d, dt, v, start, n_veh)。带磁盘缓存。"""
    cache = DATA / f"_segments_cache_{CITY}.npz"
    legacy_cache = DATA / "_segments_cache.npz"
    if CITY == "guangzhou" and (not cache.exists()) and legacy_cache.exists():
        cache = legacy_cache
    if cache.exists():
        try:                                                   # 容错：缓存损坏(如写盘被中断)则重建
            z = np.load(cache, allow_pickle=False)
            out = {k: z[k] for k in z.files}; z.close(); return out
        except Exception:
            pass
    b = BBOX
    lf = (pl.scan_parquet(str(DATA / TRACE_FILE))
          .select(["vehicle_id", pl.col("lon").cast(pl.Float64), pl.col("lat").cast(pl.Float64),
                   pl.col("speed_kmh").cast(pl.Float64), "gps_time"])
          .filter(pl.col("lon").is_between(b["lon_min"], b["lon_max"]) &
                  pl.col("lat").is_between(b["lat_min"], b["lat_max"]))
          .with_columns((pl.col("gps_time").str.slice(11, 2).cast(pl.Int32, strict=False) * 3600
                         + pl.col("gps_time").str.slice(14, 2).cast(pl.Int32, strict=False) * 60
                         + pl.col("gps_time").str.slice(17, 2).cast(pl.Int32, strict=False)).alias("t"))
          .filter(pl.col("t").is_not_null() & (pl.col("t") >= 0))
          .filter(pl.col("lon").count().over("vehicle_id") >= MIN_TRACK_POINTS)
          .with_columns(pl.col("vehicle_id").cast(pl.Categorical).to_physical().alias("vc"))
          .sort(["vc", "t"]).select(["vc", "lon", "lat", "speed_kmh", "t"]))
    df = lf.collect()
    vc = df["vc"].to_numpy().astype(np.int32)
    lon = df["lon"].to_numpy(); lat = df["lat"].to_numpy()
    spd = df["speed_kmh"].to_numpy(); t = df["t"].to_numpy().astype(np.float64)
    n = len(vc)
    start = np.empty(n, bool); start[0] = True; start[1:] = vc[1:] != vc[:-1]
    d = np.zeros(n); dt = np.zeros(n); v = np.zeros(n)
    d[1:] = haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])
    dt[1:] = (t[1:] - t[:-1]) / 3600.0
    v[1:] = 0.5 * (spd[1:] + spd[:-1])
    d[start] = 0; dt[start] = 0; v[start] = 0
    np.clip(d, 0, SEG_D_CLIP, out=d); np.clip(dt, 0, SEG_DT_CLIP, out=dt)
    out = dict(vc=vc.astype(np.int32), lon=lon.astype(np.float32), lat=lat.astype(np.float32),
               d=d.astype(np.float32), dt=dt.astype(np.float32), v=v.astype(np.float32),
               start=start, n_veh=np.int64(vc.max() + 1))
    try:
        np.savez(cache, **out)                                 # 写盘失败不阻断（下次重建）
    except Exception:
        pass
    return out


# ============================================================================
# 3. 能耗 → SoC → 低电量需求面
# ============================================================================
def cumulative_kwh(d, dt, v, start):
    """逐段能耗累加成“每辆车从自身起点起”的累计耗电 kWh。"""
    seg = d * (K_D + K_V2 * v * v) + dt * K_T
    csum = np.cumsum(seg)
    reset = np.zeros(len(seg)); idx = np.where(start)[0]
    reset[idx[1:]] = csum[idx[1:] - 1]
    return csum - np.maximum.accumulate(reset)


def low_soc_events(cum, vc, lon, lat, soc0_per_veh):
    """每辆车每个放电周期中 SoC 首次跌破 SOC_LOW 的点 → (lon, lat)。"""
    ce = cum + ((1.0 - soc0_per_veh) * BATT)[vc]                # 含初始亏电
    soc = 1.0 - (ce - np.floor(ce / BATT) * BATT) / BATT
    cyc = np.floor(ce / BATT).astype(np.int64)
    m = soc <= SOC_LOW
    key = (vc.astype(np.int64) * 100003 + cyc)[m]
    _, first = np.unique(key, return_index=True)               # 每(车,周期)取首次
    return lon[m][first], lat[m][first]


def example_soc_curves(percentiles=(30, 50, 70, 90), soc0=None):
    """取若干辆“代表车”的 SoC 曲线示例：按一天总耗电的分位数选车（30/50/70/90%）。
    返回 list[dict(veh, pct, cum_km, soc)]。soc0 默认用初始电量分布均值，便于展示。"""
    z = load_segments()
    vc, d = z["vc"], z["d"].astype(float)
    cum = cumulative_kwh(z["d"], z["dt"], z["v"], z["start"])
    n_veh = int(z["n_veh"]); last = np.zeros(n_veh)
    np.maximum.at(last, vc, cum)                       # 每车一天总耗电
    soc0 = SOC0["mean"] if soc0 is None else soc0
    pos = last > 0; out = []
    for p in percentiles:
        thr = np.percentile(last[pos], p)
        v = int(np.where(pos)[0][np.argmin(np.abs(last[pos] - thr))])
        m = vc == v
        ce = cum[m] + (1.0 - soc0) * BATT
        soc = 1.0 - (ce - np.floor(ce / BATT) * BATT) / BATT
        out.append(dict(veh=v, pct=p, cum_km=np.cumsum(d[m]), soc=soc))
    return out


def demand_surface(draws=None, seed0=1000):
    """对初始 SoC 蒙特卡洛抽样，聚合成期望需求面。
    返回 dict(lon_c, lat_c, w)：需求格中心与期望低电量需求量 w_i。"""
    draws = N_ENSEMBLE if draws is None else int(draws)
    z = load_segments()
    vc, lon, lat, n_veh = z["vc"], z["lon"], z["lat"], int(z["n_veh"])
    cum = cumulative_kwh(z["d"], z["dt"], z["v"], z["start"])
    blocks = []
    for m in range(draws):
        rng = np.random.default_rng(seed0 + m)
        s0 = np.clip(rng.normal(SOC0["mean"], SOC0["sd"], n_veh), SOC0["lo"], SOC0["hi"])
        el, ea = low_soc_events(cum, vc, lon, lat, s0)
        bid = to_block(el, ea)
        ub, cnt = np.unique(bid, return_counts=True)
        blocks.append((ub, cnt.astype(float)))
    master = np.unique(np.concatenate([u for u, _ in blocks]))
    w = np.zeros(len(master))
    for ub, cnt in blocks:
        w[np.searchsorted(master, ub)] += cnt
    w /= draws
    lc, ac = block_centroid(master)
    return dict(lon_c=lc, lat_c=ac, w=w, n_cells=len(master))


# ============================================================================
# 4. 距离矩阵
# ============================================================================
def dist_matrix(blon, blat, slon, slat, chunk=512):
    """需求格→站点的绕行距离矩阵 c[i,j] = haversine·DETOUR (km)。"""
    nD = len(blon); c = np.empty((nD, len(slon)))
    for s in range(0, nD, chunk):
        e = min(s + chunk, nD)
        c[s:e] = haversine_km(blon[s:e, None], blat[s:e, None], slon[None, :], slat[None, :]) * DETOUR
    return c


# ============================================================================
# 5. 三个错配指数
# ============================================================================
def M_old(w, c):
    """式(1) 旧指数：最近站硬指派、无限容量。仅作零拥挤极限参照。返回 需求·km。"""
    mc = c.min(axis=1)
    cov = mc <= C_BAR
    return float((w * mc * cov).sum() + P_DEAD * (w * (~cov)).sum())


def effective_capacity(fast, slow, gamma=GAMMA):
    """式(2) 有效容量 κ_j = γ·fast + slow。"""
    return gamma * np.asarray(fast, float) + np.asarray(slow, float)


def erlang_c_wait(lam, c_servers, mu):
    """式(6) M/M/c 平均排队等待 W_q（与 lam、mu 同时间单位）。
    Erlang-B 递推 B(k)=a·B/(k+a·B) 求到 B(c)，转 Erlang-C，再得 W_q；ρ≥1 返回 inf。"""
    lam = np.asarray(lam, float); c_servers = np.asarray(c_servers, np.int64); mu = np.asarray(mu, float)
    a = np.where(mu > 0, lam / np.where(mu > 0, mu, 1.0), 0.0)
    rho = a / np.maximum(c_servers, 1)
    B = np.ones_like(a); cmax = int(c_servers.max()) if c_servers.size else 0
    for k in range(1, cmax + 1):
        B = np.where(k <= c_servers, a * B / (k + a * B), B)
    denom = 1.0 - rho * (1.0 - B)
    Cw = np.where(denom > 1e-12, B / denom, 1.0)
    stable = (rho < 1.0) & (c_servers >= 1) & (mu > 0)
    return np.where(stable, Cw / (np.maximum(c_servers * mu, 1e-12) * np.maximum(1.0 - rho, 1e-12)), np.inf)


def _disp_scale(w, fast, slow, u=SYS_UTIL, gamma=GAMMA):
    """式(4) 固定容量标尺 s = Σw/(u·Σκ)。选站时固定在基线站集，避免增删站改写幸存站容量。"""
    tot_k = float(effective_capacity(fast, slow, gamma).sum())
    return (float(np.sum(w)) / (u * tot_k)) if (tot_k > 0 and u > 0) else None


def M_disp(w, c, fast, slow, u=SYS_UTIL, gamma=GAMMA, s=None, knn=DISP_LP_KNN):
    """式(3)(4) 空间置换：容量受限运输 LP（scipy HiGHS 精确解）。返回 dict(M, M_access,
    M_dead_range, M_dead_cap, load, cap)。单位 需求·km。拥堵=被挤去更远的站。
    s=None 时按当前站集现算容量标尺；做增删站分析必须传入固定 s。"""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    w = np.asarray(w, float); c = np.asarray(c, float); nD, nF = c.shape
    kappa = effective_capacity(fast, slow, gamma)
    if s is None:
        s = _disp_scale(w, fast, slow, u, gamma)
    cap = (s if s is not None else np.inf) * kappa
    reach = c <= C_BAR; has = reach.any(axis=1)
    # 弧：每格连最近 knn 个可达站（全向量化，避免 Python 循环——这是选站提速的关键）
    masked = np.where(reach, c, np.inf)
    k = int(min(knn, nF))
    idx = np.argpartition(masked, k - 1, axis=1)[:, :k]             # 每格最近 k 个站
    rows = np.repeat(np.arange(nD), k); cols = idx.ravel()
    dval = masked[rows, cols]; ok = np.isfinite(dval)              # 只留可达弧（去掉 inf 占位）
    ai = rows[ok].astype(np.int64); aj = cols[ok].astype(np.int64)
    acost = w[ai] * c[ai, aj]
    nA = len(acost)
    cost = np.concatenate([acost, P_DEAD * w])                     # x_ij 后接 dead_i
    Aeq = coo_matrix((np.ones(nA + nD), (np.concatenate([ai, np.arange(nD)]),
                      np.concatenate([np.arange(nA), nA + np.arange(nD)]))), shape=(nD, nA + nD))
    Aub = coo_matrix((w[ai], (aj, np.arange(nA))), shape=(nF, nA + nD))
    r = linprog(cost, A_ub=Aub, b_ub=cap, A_eq=Aeq, b_eq=np.ones(nD), bounds=(0, 1), method="highs")
    if not r.success:
        raise RuntimeError("空间置换 LP 未收敛：" + r.message)
    x = r.x[:nA]; dead = r.x[nA:]
    M_access = float((np.asarray(acost) * x).sum())
    load = np.zeros(nF); np.add.at(load, aj, w[ai] * x)
    M_dr = float(P_DEAD * (w[~has] * dead[~has]).sum())            # 可达内无站
    M_dc = float(P_DEAD * (w[has] * dead[has]).sum())              # 可达有站但容量满
    return dict(M=M_access + M_dr + M_dc, M_access=M_access, M_dead_range=M_dr,
                M_dead_cap=M_dc, load=load, cap=cap)


def M_queue(w, c, fast, slow):
    """式(5)(6)(7) 时间排队：最近站指派 + M/M/c 排队（偏悲观参照）。返回 dict(M, M_access,
    M_dead_range, M_dead_cap, load, Wq_min)。单位 需求·min。拥堵=原地排队。"""
    w = np.asarray(w, float); c = np.asarray(c, float); nD, nF = c.shape
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    servers = np.maximum((fast + slow).astype(np.int64), 1)
    ports = np.maximum(fast + slow, 1e-9)
    mu_eff = (fast * MU_FAST + slow * MU_SLOW) / ports             # 等效单台服务率 辆/h
    masked = np.where(c <= C_BAR, c, np.inf)
    nearest = masked.argmin(axis=1); dmin = masked[np.arange(nD), nearest]
    has = np.isfinite(dmin)
    load = np.zeros(nF); np.add.at(load, nearest[has], w[has])
    lam = load / max(OPER_HOURS, 1e-9)
    Wq = erlang_c_wait(lam, servers, mu_eff) * 60.0               # 分钟
    daily_cap = servers * mu_eff * OPER_HOURS
    over_frac = np.where(load > 1e-12, np.maximum(load - daily_cap, 0) / np.maximum(load, 1e-12), 0.0)
    wait = np.minimum(Wq, WAIT_BALK_MIN)                         # 等待按耐心上限封顶
    travel = dmin * (60.0 / AVG_SPEED); dead_min = P_DEAD * (60.0 / AVG_SPEED)
    served = np.ones(nD); served[has] = 1.0 - over_frac[nearest[has]]
    wq_block = np.zeros(nD); wq_block[has] = wait[nearest[has]]
    M_access = float((w[has] * (travel[has] + wq_block[has]) * served[has]).sum())
    M_dc = float((w[has] * (1.0 - served[has]) * dead_min).sum())  # 超当日服务能力 → 抛锚
    M_dr = float((w[~has] * dead_min).sum())                      # 可达内无站
    return dict(M=M_access + M_dc + M_dr, M_access=M_access, M_dead_range=M_dr,
                M_dead_cap=M_dc, load=load, Wq_min=Wq)


def both_indices(w, c, fast, slow):
    """同时给出三个指标，便于基线并列报告（§5.3）。空间置换折分钟以可比。"""
    disp = M_disp(w, c, fast, slow); queue = M_queue(w, c, fast, slow)
    return dict(M_old_km=M_old(w, c),
                M_disp_km=disp["M"], M_disp_min=disp["M"] * 60.0 / AVG_SPEED,
                M_queue_min=queue["M"], disp=disp, queue=queue)


# ============================================================================
# 6. 容量感知选站：S1 只增 / S2 只减 / S3 等量调配（直接优化容量口径）
# ============================================================================
def M_disp_fill(w, c, fast, slow, s=None, u=SYS_UTIL, gamma=GAMMA):
    """空间置换的**快速就近贪心填充近似**（无 LP）：按“离最近站越近的格越先占”把需求灌进
    最近的有空容量站，满了往次近溢出，可达内全满计抛锚。它比精确 LP 略高、但快上一两个数量级，
    **专用于选站(S1/S2/S3)的边际评估**——选站本就是贪心启发式。基线/交叉打分仍用精确 M_disp。"""
    w = np.asarray(w, float); c = np.asarray(c, float); nD, nF = c.shape
    kappa = effective_capacity(fast, slow, gamma)
    if s is None:
        s = _disp_scale(w, fast, slow, u, gamma)
    cap = (s if s is not None else np.inf) * kappa
    rem = cap.copy(); load = np.zeros(nF)
    # 每格只看最近 knn 个可达站（向量化预算），内层循环只跑 knn 次 → O(nD·knn)，很快
    masked = np.where(c <= C_BAR, c, np.inf)
    k = int(min(DISP_LP_KNN, nF))
    ki = np.argpartition(masked, k - 1, axis=1)[:, :k]
    kd = np.take_along_axis(masked, ki, axis=1)
    o = np.argsort(kd, axis=1)
    ki = np.take_along_axis(ki, o, axis=1); kd = np.take_along_axis(kd, o, axis=1)
    nd = kd[:, 0]; has = np.isfinite(nd)
    Ma = Mdr = Mdc = 0.0
    for i in np.argsort(np.where(has, nd, np.inf)):           # 近站优先
        if not has[i]:
            Mdr += w[i] * P_DEAD; continue
        need = w[i]
        for t in range(k):
            dj = kd[i, t]
            if not np.isfinite(dj):
                break
            j = ki[i, t]; take = rem[j]
            if take <= 1e-12:
                continue
            if need <= 1e-12:
                break
            p = need if need < take else take
            Ma += p * dj; rem[j] -= p; load[j] += p; need -= p
        if need > 1e-12:
            Mdc += need * P_DEAD
    return dict(M=Ma + Mdr + Mdc, M_access=Ma, M_dead_range=Mdr, M_dead_cap=Mdc, load=load, cap=cap)


def _eval(w, c, fast, slow, lens, s):
    """选站用的快速口径评估：disp 用就近贪心填充近似，queue 用 M/M/c（本就快）。"""
    r = M_queue(w, c, fast, slow) if lens == "queue" else M_disp_fill(w, c, fast, slow, s=s)
    return r["M"], r


def _add_pool(w, c, cc, fast, slow, s, pool_n):
    """候选站池：兼顾“够不着(距离缺口)”与“站满了(容量缺口)”两类加站机会，取各 top 一半。"""
    min_c = c.min(axis=1)
    red = (w[:, None] * np.maximum(min_c[:, None] - cc, 0.0)).sum(axis=0)        # 距离缺口
    base = M_disp_fill(w, c, fast, slow, s=s)
    util = np.where(base["cap"] > 1e-12, base["load"] / np.maximum(base["cap"], 1e-12), 0.0)
    reach0 = np.where(c <= C_BAR, c, np.inf); near = reach0.argmin(axis=1)
    cell_util = np.where(np.isfinite(reach0.min(axis=1)), util[near], 0.0)
    capsc = (w[:, None] * cell_util[:, None] * (cc <= C_BAR)).sum(axis=0)        # 容量缺口
    half = max(1, pool_n // 2)
    pool = np.unique(np.concatenate([np.argsort(red)[::-1][:half], np.argsort(capsc)[::-1][:half]]))
    sc = red / (red.max() + 1e-12) + capsc / (capsc.max() + 1e-12)
    if len(pool) < min(pool_n, cc.shape[1]):
        # 两类 top 候选可能高度重合；必须用综合得分继续回填，否则 S1/S3 会加不满目标 N。
        order = np.argsort(sc)[::-1]
        seen = set(pool.tolist())
        extra = [int(i) for i in order if int(i) not in seen]
        pool = np.concatenate([pool, np.asarray(extra[:pool_n - len(pool)], dtype=int)])
    if len(pool) > pool_n:
        pool = pool[np.argsort(sc[pool])[::-1][:pool_n]]
    return pool


def greedy_add(w, c, fast, slow, cand_lon, cand_lat, blon, blat, n_max, lens="disp",
               new_fast=None, new_slow=None, pool_n=CAP_POOL, s=None, desc=None, force_n=False):
    """S1：每步加一个让 lens 口径错配降得最多的候选站（建在需求格中心、按现有站桩数中位数赋容量）。
    惰性贪心(CELF)控制评估次数。返回 dict(sel=候选索引, M=每步后该口径 M)。"""
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    if s is None:
        s = _disp_scale(w, fast, slow)
    nf = float(max(1, round(np.median(fast)))) if new_fast is None else new_fast
    ns = float(max(0, round(np.median(slow)))) if new_slow is None else new_slow
    cc_all = haversine_km(blon[:, None], blat[:, None], cand_lon[None, :], cand_lat[None, :]) * DETOUR
    pool = _add_pool(w, c, cc_all, fast, slow, s, pool_n)
    cc = cc_all[:, pool]; pf = np.full(len(pool), nf); ps = np.full(len(pool), ns)
    cur_c, cur_f, cur_s = c, fast, slow
    cur_M, _ = _eval(w, cur_c, cur_f, cur_s, lens, s); hist = [cur_M]
    avail = np.ones(len(pool), bool); gain = np.full(len(pool), np.inf); fresh = np.full(len(pool), -1); sel = []
    target = int(n_max)
    for step in tqdm(range(target), desc=desc or f"S1 加站[{lens}]", leave=False):
        if not avail.any():
            hist.extend([cur_M] * (target - step))
            break
        while True:                                              # CELF：只重算可能最优的候选
            ids = np.where(avail)[0]; k = int(ids[np.argmax(gain[ids])])
            if fresh[k] == step:
                break
            caug = np.concatenate([cur_c, cc[:, k:k + 1]], axis=1)
            fa = np.concatenate([cur_f, pf[k:k + 1]]); sa = np.concatenate([cur_s, ps[k:k + 1]])
            gain[k] = cur_M - _eval(w, caug, fa, sa, lens, s)[0]; fresh[k] = step
        if (gain[k] <= 1e-9) and (not force_n):
            # 没有正收益时不再新增，但把后续政策规模补成平线，避免图上点数少于 ADD_FRACS。
            hist.extend([cur_M] * (target - step))
            break
        cur_c = np.concatenate([cur_c, cc[:, k:k + 1]], axis=1)
        cur_f = np.concatenate([cur_f, pf[k:k + 1]]); cur_s = np.concatenate([cur_s, ps[k:k + 1]])
        cur_M -= gain[k]; avail[k] = False; sel.append(int(pool[k])); hist.append(cur_M)
    return dict(sel=sel, M=hist, new_fast=nf, new_slow=ns)


def greedy_remove(w, c, fast, slow, n_remove, lens="disp", s=None):
    """S2：每步删一个当前该口径下“负载最低(最像冗余)”的站，重算后继续。
    容量口径下负载≈0 的站删掉几乎不增成本——这才是真正的容量冗余。返回 dict(order, M)。"""
    fast = np.asarray(fast, float); slow = np.asarray(slow, float); nF = c.shape[1]
    if s is None:
        s = _disp_scale(w, fast, slow)
    keep = np.ones(nF, bool); idx = np.arange(nF); order = []
    M0, info = _eval(w, c, fast, slow, lens, s); hist = [M0]
    for _ in tqdm(range(int(min(max(n_remove, 0), nF - 1))), desc=f"S2 删站[{lens}]", leave=False):
        cols = idx[keep]
        if cols.size < 2:
            break
        j = int(np.argmin(info["load"]))
        keep[cols[j]] = False; order.append(int(cols[j]))
        Mk, info = _eval(w, c[:, keep], fast[keep], slow[keep], lens, s); hist.append(Mk)
    return dict(order=order, M=hist)


def swap(w, c, fast, slow, cand_lon, cand_lat, blon, blat, n_swap, lens="disp", pool_n=CAP_POOL, s=None):
    """S3 等量调配：**先**贪心找出最多 n_swap 个有价值的新增站(可能因收敛而少于 n_swap)，
    **再**删掉**同样数量**的最低负载站——保证关、开数量严格相等(net 站数不变)。
    返回 dict(removed, added, M)；removed 与 added 等长。"""
    fast = np.asarray(fast, float); slow = np.asarray(slow, float)
    if s is None:
        s = _disp_scale(w, fast, slow)
    target = int(max(0, min(n_swap, c.shape[1] - 1, len(cand_lon))))
    add = greedy_add(w, c, fast, slow, cand_lon, cand_lat, blon, blat, target,
                     lens, pool_n=pool_n, s=s, force_n=True)
    rm = greedy_remove(w, c, fast, slow, target, lens, s)["order"] if target > 0 else []
    added = add["sel"][:target]
    rm = rm[:len(added)]
    # 逐个规模评估“关闭前 n 个 + 新增前 n 个”的最终布局，保证 S3 曲线和地图都对应真实等量调配。
    hist = []
    base_M, _ = _eval(w, c, fast, slow, lens, s)
    hist.append(base_M)
    nf, ns = add["new_fast"], add["new_slow"]
    cc_all = haversine_km(blon[:, None], blat[:, None], cand_lon[None, :], cand_lat[None, :]) * DETOUR
    for k in range(1, len(added) + 1):
        keep = np.ones(c.shape[1], bool)
        keep[np.asarray(rm[:k], int)] = False
        add_cols = cc_all[:, np.asarray(added[:k], int)]
        c2 = np.concatenate([c[:, keep], add_cols], axis=1)
        f2 = np.concatenate([fast[keep], np.full(k, nf)])
        s2 = np.concatenate([slow[keep], np.full(k, ns)])
        hist.append(_eval(w, c2, f2, s2, lens, s)[0])
    return dict(removed=rm, added=added, M=hist, new_fast=nf, new_slow=ns)


def add_counts(nF):
    """新增/调配规模 = 现有站数的 1/2/5/10%，至少 1。"""
    from math import ceil
    return tuple(max(1, int(ceil(nF * f))) for f in ADD_FRACS)
