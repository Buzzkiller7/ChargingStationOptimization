"""
cso_config.py — 中央配置：城市注册表 + 物理/模型参数（多城市可复用）：
  * 城市相关的一切（数据文件、经纬度边界、站点表列名、行政区划码）都集中在
    CITIES 注册表里；新增一个城市只需在 CITIES 里加一条。
  * 物理/模型参数（能耗、阈值、成本、栅格、集成次数）是城市无关的全局默认，
    个别城市如需覆盖可在其条目里加同名键（引擎读取时优先取城市级）。
  * 路径默认相对“本文件上一级目录”（即 Code/ 的上一级 = 项目根，原始数据所在）。
    可用环境变量 CSO_DATA / CSO_CACHE 覆盖；用 CSO_CITY 选择当前城市。

"""
from __future__ import annotations
import os
from math import ceil
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # Code/
DATA_ROOT = Path(os.environ.get("CSO_DATA", HERE.parent))    # 原始数据目录（默认=项目根）
CACHE = Path(os.environ.get("CSO_CACHE", HERE / "_cache"))   # 中间数组缓存
CACHE.mkdir(parents=True, exist_ok=True)

# ============================ 城市注册表============================
CITIES = {
    "guangzhou": dict(
        label="广州 Guangzhou",
        raw="Taxi_2019_10_14_admin_4401.parquet",           # 按三级区划码筛好的轨迹
        trace_date="2019-10-14",
        stations="guangzhou_station.csv",
        bbox=dict(lon_min=112.90, lon_max=114.10, lat_min=22.50, lat_max=24.00),
        station_cols=dict(lon="WGS84_station_lg", lat="WGS84_station_lt",
                          fast="station_fast_cnt", slow="station_slow_cnt",
                          create_time="create_time"), # 站点表经纬度列名、快慢充数量、建成时间列名
        admin_prefix="4401",                                 # 从全国数据筛选广州用
    ),
    # ---- 新增城市模板 ----
    # "shenzhen": dict(
    #     label="深圳 Shenzhen",
    #     raw="Taxi_2019_10_14_admin_4403.parquet",
    #     trace_date="2019-10-14",
    #     stations="shenzhen_station.csv",
    #     bbox=dict(lon_min=113.70, lon_max=114.70, lat_min=22.40, lat_max=22.90),
    #     station_cols=dict(lon="WGS84_station_lg", lat="WGS84_station_lt",
    #                       fast="station_fast_cnt", slow="station_slow_cnt",
    #                       create_time="create_time"),
    #     admin_prefix="4403",
    # ),
}
ACTIVE_CITY = os.environ.get("CSO_CITY", "guangzhou")


def use_city(name: str):
    """切换当前城市"""
    global ACTIVE_CITY
    if name not in CITIES:
        raise KeyError(f"未注册的城市 '{name}'，可选: {list(CITIES)}")
    ACTIVE_CITY = name
    return city()


def city():
    """返回当前城市配置"""
    c = dict(CITIES[ACTIVE_CITY])
    c["key"] = ACTIVE_CITY
    c["raw_path"] = DATA_ROOT / c["raw"]
    c["stations_path"] = DATA_ROOT / c["stations"]
    return c


def cache_file(kind: str) -> Path:
    """城市隔离的缓存文件名，避免多城市互相覆盖。"""
    return CACHE / f"{kind}_{ACTIVE_CITY}.npz"


# ============================ 空间栅格 ============================
GRID_DEG = 0.01            # 0.01° ≈ 1 km
ENC = 10001                # block_id = row*ENC + col 的编码基

# ============================ 车型能耗模型 ============================
# 能耗ΔE = d·(k_d + k_v2·v²) + Δt·k_t   [d km, v km/h, Δt s]
# SoC = 1 − (累计耗电 mod batt)/batt（满电瞬时重置）
VEHICLES = {
    "model3": dict(batt=78.4, k_d=0.150, k_v2=0.000025, k_t=0.00025 / 3600),   # Tesla Model3 LR
    "byd60":  dict(batt=60.0, k_d=0.120, k_v2=0.000020, k_t=0.00025 / 3600),   # 主流国产网约车~60kWh
}
VEH_DEFAULT = "model3"

# ============================ 需求定义 ============================
SOC_LOW = 0.20             # 低电量触发阈值
SEG_DT_CLIP = 1800.0       # 截断 Δt 抑制 GPS 断点 (s)
SEG_D_CLIP = 50.0          # 截断 Δd 抑制 GPS 跳点 (km)
# 初始 SoC：v0 固定全员 1.0（最大的自由参数）；本版作为分布并蒙特卡洛集成
SOC0_TNORM = dict(mean=0.85, sd=0.10, lo=0.40, hi=1.00)        # 截断正态，l

# ============================ 成本 / 目标函数 ============================
# c_ij = haversine(i,j) · DETOUR
# M(F) = Σ_i w_i·min_j c_ij·1[min c ≤ C_BAR] + P_DEAD·Σ_i w_i·1[min c > C_BAR]
DETOUR = 1.3               # 考虑道路曲折、交通管制等因素的路径偏离系数
C_BAR = 100.0              # 20%→0% 可达里程 (km)；这里取Model3的近似值 C_BAR≈SOC_LOW·batt/k_d
P_DEAD = 1000.0            # 抛锚惩罚

# ============================ 集成 / 重采样 ============================
# 在初始 SoC 分布上蒙特卡洛抽样，得到多个需求实例；新增站情景按现有站点数的固定比例定义，便于全国多城市横向比较。
N_ENSEMBLE = 100            # 初始 SoC 蒙特卡洛抽样数
N_BOOT = 150               # 自助重采样数
ADD_STATION_FRACS = (0.01, 0.02, 0.05, 0.1)  # 新增站情景：现有站点数的 1%、2%、5%、10%
CAND_CAP = 1000000         # 候选格上限；设为很大时等价于使用全部有需求格作为新增候选
SEED = 25


def add_station_counts(n_existing: int) -> tuple[int, ...]:
    """把新增站比例转换成该城市的新增站数量，至少新增 1 个。"""
    return tuple(max(1, int(ceil(n_existing * frac))) for frac in ADD_STATION_FRACS)
