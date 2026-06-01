"""
cso_engine.py — 网约车轨迹充电站优化的核心计算引擎。

本文件把“原始网约车 GPS 轨迹”和“已有充电站表”转换成可以做选址优化的数学对象：
1. 读取 cso_config.py 中当前城市 ACTIVE_CITY 的轨迹、站点、边界和参数。
2. 将 GPS 点按车辆和时间排序，预计算相邻轨迹点之间的距离/时间/速度分段，并缓存到 npz。
3. 用车型能耗参数把轨迹分段转换为累计耗电量，再推算车辆 SoC 变化。
4. 找到每辆车每个放电周期中首次低于 SoC 阈值的位置，把这些位置聚合成栅格需求块。
5. 用蒙特卡洛抽样初始 SoC，生成更稳健的期望需求面。
6. 计算需求块到充电站的绕行距离矩阵，并用错配指数 M 衡量“需求到最近站”的总成本。
7. 三类情景分析：
   S1 只增站：在候选需求格中贪心新增站点。
   S2 只减/边际：分析现有站点边际价值、冗余和负载集中度。
   S3 等量调配：关闭低边际站，同时新增高价值候选站，总站数不变。

函数清单
----------------
* haversine_km：经纬度点对球面距离，支持 numpy 广播。
* to_block / block_centroid：经纬度与栅格 block_id 互转。
* precompute_segments：一次性清洗轨迹和站点，并写入城市隔离缓存。
* load_segments / load_stations：读取上述缓存。
* cumulative_kwh：按车型能耗模型，把分段轨迹累加成每车累计耗电。
* low_soc_events：提取每车每个放电周期首次低 SoC 事件。
* aggregate_blocks：把低 SoC 事件聚合为需求栅格及权重。
* load_admin_boundaries：读取/缓存当前城市的二级行政区边界。
* draw_city_context：在 Matplotlib 图上统一叠加行政边界、规则栅格和城市视野。
* district_distribution：按二级行政区统计 GPS 记录、车辆和站点供给分布。
* tnorm：抽样每辆车的初始 SoC。
* naive_demand_surface：全员满电初始 SoC 的基线需求面。
* monte_carlo_demand_surface / demand_surface：蒙特卡洛集成，输出期望需求面。
* align_demand_surfaces：把 naive 与蒙特卡洛需求面并到同一组栅格，便于比较。
* block_station_dist：生成需求块到站点的距离矩阵。
* compute_M：根据最近站距离和抛锚惩罚计算错配指数。
* greedy_add：S1，只新增站点的贪心优化。
* marginal_delta：S2，计算关闭某站带来的边际损失。
* pigeonhole_decomposition：拆解零边际站比例的结构性成分。
* event_load：统计每个站承接的需求负载和集中度。
* swap：S3，等量关闭/新增站点的调配优化。

主要输入
--------
* cso_config.py：城市注册表、bbox、路径、车型能耗参数、SoC 阈值、目标函数参数。
* 原始轨迹 parquet：至少包含 vehicle_id、lon、lat、speed_kmh、gps_time。
* 站点 csv：列名由 cso_config.CITIES[city]["station_cols"] 映射为 lon/lat/fast/slow。
* 下游优化函数输入：需求块经纬度、需求权重 w、现有站经纬度、候选站经纬度、距离矩阵等。

主要输出
--------
* Code/_cache/segments_<city>.npz：车辆编码 vc、GPS lon/lat、分段距离 d、时间 dt、速度 v、车辆起点 start、车辆数 n_veh。
* Code/_cache/stations_<city>.npz：清洗后的站点 lon/lat/fast/slow。
* naive_demand_surface 返回 dict：master 栅格、lon_c/lat_c、naive 需求 w_naive、事件数 n_ev、需求块数 nD。
* monte_carlo_demand_surface 返回 dict：master 栅格、lon_c/lat_c、期望需求 w_exp、出现频率 appear、每次抽样 reals 等。
* align_demand_surfaces 返回 dict：共同 master 栅格上对齐后的 w_exp、w_naive、appear 及原始抽样元数据。
* block_station_dist 返回 c[i, j]：需求块 i 到站点 j 的绕行距离。
* compute_M 返回错配指数 M；greedy_add / marginal_delta / event_load / swap 返回三类情景分析结果。

"""
from __future__ import annotations
import json
import re
import urllib.request
import warnings
import numpy as np
import polars as pl
import pandas as pd
import cso_config as C


# ------------------------------- 几何 / 栅格 ----------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    # 计算经纬度点对之间的 haversine 距离（km）。
    R = 6371.0088
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# 把经纬度落到城市 bbox 内的规则栅格，并编码成整数 block_id。
def to_block(lon, lat, grid=None):
    grid = grid or C.GRID_DEG; b = C.city()["bbox"]
    ii = np.clip(((lat - b["lat_min"]) / grid).astype(np.int32), 0, 100000)
    jj = np.clip(((lon - b["lon_min"]) / grid).astype(np.int32), 0, 100000)
    return ii.astype(np.int64) * C.ENC + jj


# 把 block_id 解码回栅格中心经纬度。
def block_centroid(block_id, grid=None):
    grid = grid or C.GRID_DEG; b = C.city()["bbox"]
    ii = block_id // C.ENC; jj = block_id % C.ENC
    return (b["lon_min"] + (jj + 0.5) * grid).astype(np.float64), \
           (b["lat_min"] + (ii + 0.5) * grid).astype(np.float64)


# ------------------------------- 地图底图 / 行政区 ----------------------------------
# 从城市配置 label 中抽取中文城市名，供在线行政区目录匹配使用。
def _city_name_from_cfg(cfg=None):
    cfg = cfg or C.city()
    label = str(cfg.get("label") or cfg.get("key") or "")
    m = re.search(r"[\u4e00-\u9fff]+", label)
    return (m.group(0).removesuffix("市") if m else label.split()[0]).strip()


# 带缓存的 JSON 下载/读取。行政区目录和 GeoJSON 都存在 C.CACHE 下，避免重复联网。
def _download_json(url, path, timeout=20):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            path.write_bytes(resp.read())
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v0 in obj.values():
            yield from _walk_dicts(v0)
    elif isinstance(obj, list):
        for v0 in obj:
            yield from _walk_dicts(v0)


def _six_digit_codes(value):
    return re.findall(r"(?<!\d)(\d{6})(?!\d)", str(value))


def _best_admin_code(props, prefix=""):
    codes = []
    for v0 in props.values():
        codes.extend(_six_digit_codes(v0))
    if prefix:
        district_codes = [c for c in codes if c.startswith(prefix) and not c.endswith("00")]
        if district_codes:
            return district_codes[0]
        city_codes = [c for c in codes if c.startswith(prefix)]
        if city_codes:
            return city_codes[0]
    return codes[0] if codes else None


def _best_place_name(props, fallback):
    names = []
    for v0 in props.values():
        s = str(v0).strip()
        if re.search(r"[\u4e00-\u9fff]", s) and not re.search(r"\d{4,}", s) and len(s) <= 30:
            names.append(s)
    preferred = [s for s in names if any(k in s for k in ["区", "县", "市", "旗"])]
    return (preferred or names or [fallback])[0]


# 根据城市中文名和 admin_prefix 找市级 adcode；失败时退回 prefix + "00"。
def resolve_city_adcode(cfg=None):
    cfg = cfg or C.city()
    city_name = _city_name_from_cfg(cfg)
    admin_prefix = str(cfg.get("admin_prefix", "") or "")
    urls = [
        "https://geo.datav.aliyun.com/areas_v3/bound/infos.json",
        "https://geo.datav.aliyun.com/areas_v3/bound/all.json",
    ]
    for k, url in enumerate(urls):
        try:
            data = _download_json(url, C.CACHE / f"admin_area_catalog_{k}.json")
        except Exception:
            continue
        for item in _walk_dicts(data):
            joined_values = " ".join(str(v0) for v0 in item.values())
            if city_name and city_name in joined_values:
                codes = []
                for key, value in item.items():
                    codes.extend(_six_digit_codes(key))
                    codes.extend(_six_digit_codes(value))
                if admin_prefix:
                    codes = [c for c in codes if c.startswith(admin_prefix)] or codes
                city_codes = [c for c in codes if c.endswith("00")] or codes
                if city_codes:
                    return city_codes[0]
    if admin_prefix:
        return admin_prefix + "00"
    raise RuntimeError(f"无法根据城市名 {city_name!r} 解析市级 adcode。")


# 读取当前城市的二级行政区边界。需要 geopandas；GeoJSON 会缓存到 Code/_cache。
def load_admin_boundaries(cfg=None, city_adcode=None, timeout=20):
    """返回当前城市二级行政区 GeoDataFrame 和说明文字。

    GeoJSON 来源为 DataV 行政区边界接口；若文件已在 C.CACHE 中则直接读缓存。
    """
    cfg = cfg or C.city()
    admin_prefix = str(cfg.get("admin_prefix", "") or "")
    city_adcode = city_adcode or resolve_city_adcode(cfg)
    admin_prefix = city_adcode[:4] if city_adcode else admin_prefix
    city_name = _city_name_from_cfg(cfg)
    try:
        import geopandas as gpd
    except Exception as exc:
        raise RuntimeError("load_admin_boundaries 需要安装 geopandas。") from exc

    admin_geojson = C.CACHE / f"admin_{city_adcode}_full.geojson"
    if not admin_geojson.exists():
        url = f"https://geo.datav.aliyun.com/areas_v3/bound/{city_adcode}_full.json"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            admin_geojson.write_bytes(resp.read())

    admin_gdf = gpd.read_file(admin_geojson).to_crs(4326)
    prop_cols = [c for c in admin_gdf.columns if c != "geometry"]
    admin_gdf["district_code"] = [
        _best_admin_code(row[prop_cols].to_dict(), admin_prefix)
        for _, row in admin_gdf.iterrows()
    ]
    admin_gdf["district_name"] = [
        _best_place_name(row[prop_cols].to_dict(), code)
        for code, (_, row) in zip(admin_gdf["district_code"], admin_gdf.iterrows())
    ]
    admin_gdf = admin_gdf.dropna(subset=["district_code", "geometry"]).copy()
    admin_gdf["district_code"] = admin_gdf["district_code"].astype(str).str.slice(0, 6)
    admin_gdf["district_name"] = admin_gdf["district_name"].fillna(admin_gdf["district_code"]).astype(str)
    admin_gdf = admin_gdf[admin_gdf["district_code"].str.startswith(admin_prefix)].copy()
    admin_gdf = admin_gdf.dissolve(by=["district_code", "district_name"], as_index=False)
    if admin_gdf.empty:
        raise RuntimeError(f"{admin_geojson.name} 未解析到 {city_name} 的二级行政区边界。")
    note = f"行政区边界：按城市名 {city_name!r} 解析为 {city_adcode}，使用 {admin_geojson.name}。"
    return admin_gdf, note


# 统一地图上下文：行政区边界 + 规则栅格 + bbox 视野。业务图层在调用后继续 scatter/plot 即可。
def draw_city_context(ax=None, cfg=None, bbox=None, grid=None, admin_gdf=None,
                      show_admin=True, show_grid=True, show_labels=False,
                      fill_admin=False, admin_fill_color="#f7f7f7",
                      boundary_color="#555555", boundary_linewidth=0.8,
                      grid_color="#1f77b4", grid_linewidth=0.3, grid_alpha=0.28,
                      label_fontsize=8, set_extent=True, xlabel="Longitude",
                      ylabel="Latitude", warn=True):
    """在 Matplotlib 坐标轴上绘制城市行政边界和需求模型栅格。

    返回 dict(ax, admin_gdf, note)。若 geopandas/网络/缓存不可用，默认只画 bbox 栅格并给出 warning。
    """
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(6.4, 6.4))
    cfg = cfg or C.city()
    b = bbox or cfg["bbox"]
    grid = grid or C.GRID_DEG
    note = ""

    if show_admin:
        try:
            if admin_gdf is None:
                admin_gdf, note = load_admin_boundaries(cfg)
            if fill_admin:
                admin_gdf.plot(ax=ax, color=admin_fill_color, edgecolor="white", linewidth=0.5, zorder=0)
            admin_gdf.boundary.plot(ax=ax, color=boundary_color, linewidth=boundary_linewidth, zorder=2)
            if show_labels and "district_name" in admin_gdf:
                for _, row in admin_gdf.iterrows():
                    p0 = row.geometry.representative_point()
                    ax.text(p0.x, p0.y, str(row["district_name"]), ha="center", va="center",
                            fontsize=label_fontsize, color="#222222", zorder=3)
        except Exception as exc:
            note = f"行政区边界未绘制：{type(exc).__name__}: {exc}"
            if warn:
                warnings.warn(note)

    if show_grid:
        lon_lines = np.arange(b["lon_min"], b["lon_max"] + grid, grid)
        lat_lines = np.arange(b["lat_min"], b["lat_max"] + grid, grid)
        ax.vlines(lon_lines, ymin=b["lat_min"], ymax=b["lat_max"],
                  color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha, zorder=1)
        ax.hlines(lat_lines, xmin=b["lon_min"], xmax=b["lon_max"],
                  color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha, zorder=1)

    if set_extent:
        ax.set_xlim(b["lon_min"], b["lon_max"])
        ax.set_ylim(b["lat_min"], b["lat_max"])
    ax.set_aspect("equal", adjustable="box")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return dict(ax=ax, admin_gdf=admin_gdf, note=note)


# 自动识别某列是否像当前城市的 6 位行政区编码。
def _admin_column_score(s, admin_prefix):
    code = s.dropna().astype(str).str.extract(r"(\d{6})", expand=False).dropna()
    if len(code) == 0:
        return 0.0
    return float(code.str.startswith(admin_prefix).mean())


def _cv(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.mean() == 0 else x.std(ddof=0) / x.mean()


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    n = len(x)
    return (2 * np.arange(1, n + 1).dot(x)) / (n * x.sum()) - (n + 1) / n


# 按二级行政区统计 GPS 活动和充电站供给，并返回可直接制图的 GeoDataFrame。
def district_distribution(lf, rows=None, cfg=None, gps_sample_n=50_000, admin_gdf=None):
    """统计当前城市各二级行政区的 GPS 记录、车辆数、站点数和枪数。

    参数
    ----
    lf : polars LazyFrame
        原始轨迹 lazy scan，需包含 vehicle_id 和某个 6 位行政区编码列。
    rows : int, optional
        轨迹总行数；传入可避免重复计算。

    返回
    ----
    dict(dist, uniformity, admin_gdf, map_gdf, station_gdf, city_adcode, admin_col, boundary_note)
    """
    cfg = cfg or C.city()
    b = cfg["bbox"]
    city_adcode = resolve_city_adcode(cfg)
    admin_prefix = city_adcode[:4]
    city_name = _city_name_from_cfg(cfg)

    gps_rows = int(rows) if rows is not None else lf.select(pl.len()).collect().item()
    sample_n = min(gps_rows, gps_sample_n)
    gps_sample = lf.gather_every(max(1, gps_rows // sample_n)).collect().to_pandas()
    admin_scores = {col: _admin_column_score(gps_sample[col], admin_prefix) for col in gps_sample.columns}
    admin_col = max(admin_scores, key=admin_scores.get)
    if admin_scores[admin_col] < 0.5:
        raise RuntimeError(f"无法自动识别 GPS 行政区编码列；候选得分: {admin_scores}")

    district_expr = pl.col(admin_col).cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("district_code")
    gps_dist = (
        lf.select([district_expr, pl.col("vehicle_id")])
          .filter(pl.col("district_code").str.starts_with(admin_prefix))
          .group_by("district_code")
          .agg([
              pl.len().alias("gps_records"),
              pl.col("vehicle_id").n_unique().alias("vehicles"),
          ])
          .collect()
          .to_pandas()
    )
    gps_dist["district_code"] = gps_dist["district_code"].astype(str)

    if admin_gdf is None:
        admin_gdf, boundary_note = load_admin_boundaries(cfg, city_adcode=city_adcode)
    else:
        boundary_note = f"行政区边界：使用传入的 admin_gdf；市级 adcode={city_adcode}。"
    district_lookup = admin_gdf[["district_code", "district_name"]].drop_duplicates()

    try:
        import geopandas as gpd
    except Exception as exc:
        raise RuntimeError("district_distribution 需要安装 geopandas。") from exc

    sc = cfg["station_cols"]
    station_work = pd.read_csv(cfg["stations_path"]).dropna(subset=[sc["lon"], sc["lat"]]).copy()
    station_work = station_work[
        station_work[sc["lon"]].between(b["lon_min"], b["lon_max"]) &
        station_work[sc["lat"]].between(b["lat_min"], b["lat_max"])
    ].reset_index(drop=True)
    if sc["slow"] not in station_work:
        station_work[sc["slow"]] = 0

    point_gdf = gpd.GeoDataFrame(
        station_work,
        geometry=gpd.points_from_xy(station_work[sc["lon"]], station_work[sc["lat"]]),
        crs=4326,
    )
    try:
        joined = gpd.sjoin(point_gdf, admin_gdf[["district_code", "geometry"]], how="left", predicate="within")
    except TypeError:
        joined = gpd.sjoin(point_gdf, admin_gdf[["district_code", "geometry"]], how="left", op="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    station_work["district_code"] = joined.reindex(station_work.index)["district_code"].astype("string").reset_index(drop=True)

    station_dist = (
        station_work.dropna(subset=["district_code"])
        .assign(_station=1)
        .groupby("district_code", as_index=False)
        .agg(stations=("_station", "sum"),
             fast_guns=(sc["fast"], "sum"),
             slow_guns=(sc["slow"], "sum"))
    )
    station_dist["district_code"] = station_dist["district_code"].astype(str)

    dist = (
        district_lookup.merge(gps_dist, on="district_code", how="left")
        .merge(station_dist, on="district_code", how="left")
    )
    dist["district_name"] = dist["district_name"].fillna(dist["district_code"])
    for col in ["gps_records", "vehicles", "stations", "fast_guns", "slow_guns"]:
        if col not in dist:
            dist[col] = 0
        dist[col] = dist[col].fillna(0).astype(int)
    dist["gps_share"] = dist["gps_records"] / max(dist["gps_records"].sum(), 1)
    dist["station_share"] = dist["stations"] / max(dist["stations"].sum(), 1)
    dist["station_per_10k_gps"] = np.where(
        dist["gps_records"] > 0,
        dist["stations"] / dist["gps_records"] * 10_000,
        np.nan,
    )
    dist["share_gap_station_minus_gps"] = dist["station_share"] - dist["gps_share"]
    dist = dist.sort_values("gps_records", ascending=False).reset_index(drop=True)

    uniformity = pd.DataFrame({
        "metric": ["GPS records", "stations"],
        "total": [dist["gps_records"].sum(), dist["stations"].sum()],
        "mean_per_district": [dist["gps_records"].mean(), dist["stations"].mean()],
        "cv": [_cv(dist["gps_records"]), _cv(dist["stations"])],
        "gini": [_gini(dist["gps_records"]), _gini(dist["stations"])],
    })

    map_gdf = admin_gdf.merge(dist.drop(columns=["district_name"]), on="district_code", how="left")
    map_gdf["district_name"] = map_gdf["district_name"].fillna(map_gdf["district_code"])
    for col in ["gps_records", "vehicles", "stations", "fast_guns", "slow_guns", "gps_share", "station_share"]:
        map_gdf[col] = map_gdf[col].fillna(0)
    station_gdf = gpd.GeoDataFrame(
        station_work,
        geometry=gpd.points_from_xy(station_work[sc["lon"]], station_work[sc["lat"]]),
        crs=4326,
    )

    return dict(
        dist=dist,
        uniformity=uniformity,
        admin_gdf=admin_gdf,
        map_gdf=map_gdf,
        station_gdf=station_gdf,
        city_adcode=city_adcode,
        city_name=city_name,
        admin_col=admin_col,
        admin_scores=admin_scores,
        boundary_note=boundary_note + " 站点分区：GeoJSON 空间匹配。",
    )


# ------------------------------- 预计算分段 ----------------------------------
# 读取当前城市原始轨迹和站点表，做边界裁剪、排序、分段距离/时间计算，
def precompute_segments():
    """读当前城市原始轨迹一次，缓存车型无关的分段数组 + 清洗后的站点表。"""
    # 城市配置 - 裁剪到城市边界 bbox 内 - 从 gps_time 提取一天中的秒数 t - 把 vehicle_id 编码成整数 vc - 按车辆 vc、时间 t 排序
    cfg = C.city(); b = cfg["bbox"] 
    lf = (
        pl.scan_parquet(str(cfg["raw_path"])) 
        .select(["vehicle_id",
                 pl.col("lon").cast(pl.Float32), pl.col("lat").cast(pl.Float32), 
                 pl.col("speed_kmh").cast(pl.Float32), "gps_time"])
        .filter(pl.col("lon").is_between(b["lon_min"], b["lon_max"]) &
                pl.col("lat").is_between(b["lat_min"], b["lat_max"]))
        .with_columns(
            (pl.col("gps_time").str.slice(11, 2).cast(pl.Int32, strict=False) * 3600 
             + pl.col("gps_time").str.slice(14, 2).cast(pl.Int32, strict=False) * 60
             + pl.col("gps_time").str.slice(17, 2).cast(pl.Int32, strict=False)).alias("t"))
        .filter(pl.col("t").is_not_null() & (pl.col("t") >= 0))
        .with_columns(pl.col("vehicle_id").cast(pl.Categorical).to_physical().alias("vc"))
        .sort(["vc", "t"]).select(["vc", "lon", "lat", "speed_kmh", "t"])
    )
    df = lf.collect()
    vc = df["vc"].to_numpy().astype(np.int32)
    lon = df["lon"].to_numpy().astype(np.float32); lat = df["lat"].to_numpy().astype(np.float32)
    spd = df["speed_kmh"].to_numpy().astype(np.float32); t = df["t"].to_numpy().astype(np.float32)
    del df, lf
    
    # 计算相邻 GPS 点之间的：d 距离 km，dt 时间差 s，v  = 平均速度 km/h；识别每辆车的第一条记录 start
    n = len(vc)
    start = np.empty(n, bool); start[0] = True; start[1:] = vc[1:] != vc[:-1]
    d = np.zeros(n, np.float32); dt = np.zeros(n, np.float32); v = np.zeros(n, np.float32)
    d[1:] = haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]).astype(np.float32)
    dt[1:] = t[1:] - t[:-1]; v[1:] = 0.5 * (spd[1:] + spd[:-1])
    d[start] = 0; dt[start] = 0; v[start] = 0

    # 分段数据裁剪（负值、异常大值），写入城市隔离缓存；站点表按城市列名映射 + 边界裁剪后写入缓存
    np.clip(d, 0, C.SEG_D_CLIP, out=d); np.clip(dt, 0, C.SEG_DT_CLIP, out=dt)
    # Columns in segments cache: vc, lon, lat, d, dt, v, start, n_veh
    np.savez(C.cache_file("segments"), vc=vc, lon=lon, lat=lat, d=d, dt=dt, v=v,
             start=start, n_veh=np.int64(vc.max() + 1))
    
    # 现有充电站预处理
    sc = cfg["station_cols"]
    st = pd.read_csv(cfg["stations_path"]).dropna(subset=[sc["lon"], sc["lat"]])
    st = st.rename(columns={sc["lon"]: "lon", sc["lat"]: "lat", sc["fast"]: "fast"})
    st = st[(st.lon.between(b["lon_min"], b["lon_max"])) & (st.lat.between(b["lat_min"], b["lat_max"]))]
    slow = st[sc["slow"]].fillna(0).to_numpy() if sc["slow"] in st else np.zeros(len(st))
    # Columns in stations cache: lon, lat, fast, slow
    np.savez(C.cache_file("stations"), lon=st["lon"].to_numpy(), lat=st["lat"].to_numpy(),
             fast=st["fast"].fillna(0).to_numpy().astype(np.int32), slow=np.asarray(slow).astype(np.int32))
    
    return dict(city=cfg["key"], rows=int(n), vehicles=int(vc.max() + 1), stations=int(len(st)))


# 读取 precompute_segments 写出的轨迹分段缓存。
def load_segments():
    return np.load(C.cache_file("segments"))


# 读取 precompute_segments 写出的站点缓存。
def load_stations():
    return np.load(C.cache_file("stations"))


# -------------------------------- SoC / 需求 ---------------------------------
# 把轨迹分段转换为“每辆车从自身起点开始”的累计耗电量(kWh)。
def cumulative_kwh(d, dt, v, vc, start, veh=None): # 
    p = C.VEHICLES[veh or C.VEH_DEFAULT]
    seg = (d * (p["k_d"] + p["k_v2"] * v * v) + dt * p["k_t"]).astype(np.float64)
    csum = np.cumsum(seg)
    reset = np.zeros(len(seg)); idx = np.where(start)[0] # 每辆车起点索引
    reset[idx[1:]] = csum[idx[1:] - 1] 
    return csum - np.maximum.accumulate(reset) # 每辆车的累计耗电 = 全局累计耗电 - 上一个起点的累计耗电


# 从累计耗电曲线中提取低电量事件。
def low_soc_events(cum, vc, lon, lat, soc0_per_vehicle, batt, thr=None):
    """每个 (车,放电周期) 中 SoC 首次跌破阈值的点 → (lon, lat)。去采样偏差。"""
    thr = C.SOC_LOW if thr is None else thr
    ce = cum + ((1.0 - soc0_per_vehicle) * batt)[vc]
    soc = 1.0 - (ce - np.floor(ce / batt) * batt) / batt
    cyc = np.floor(ce / batt).astype(np.int64)
    m = soc <= thr
    key = (vc.astype(np.int64) * 100003 + cyc)[m]
    ll = lon[m].astype(np.float64); aa = lat[m].astype(np.float64)
    _, first = np.unique(key, return_index=True)

    return ll[first], aa[first]


# 把低 SoC 事件点聚合成栅格需求。
def aggregate_blocks(ev_lon, ev_lat, grid=None):
    bid = to_block(ev_lon, ev_lat, grid)
    ub, w = np.unique(bid, return_counts=True)
    lc, ac = block_centroid(ub, grid)
    return ub, w.astype(np.float64), lc, ac


# 按配置的截断正态分布生成每辆车的初始 SoC。
def tnorm(n, rng):
    """按 SOC0_TNORM 抽取每车初始 SoC（截断正态）。"""
    p = C.SOC0_TNORM
    return np.clip(rng.normal(p["mean"], p["sd"], n), p["lo"], p["hi"])


# 读取轨迹缓存并计算后续需求建模共用的累计耗电曲线。
def _demand_context(veh=None):
    z = load_segments(); vc, lon, lat = z["vc"], z["lon"], z["lat"]
    d, dt, v, start = z["d"], z["dt"], z["v"], z["start"]; n_veh = int(z["n_veh"])
    batt = C.VEHICLES[veh or C.VEH_DEFAULT]["batt"]
    cum = cumulative_kwh(d, dt, v, vc, start, veh)
    return vc, lon, lat, n_veh, batt, cum


# 满电初始 SoC 的基线需求面。
def naive_demand_surface(veh=None, thr=None, grid=None):
    """全员初始 SoC=1.0 的基线需求面。
    return dict(master, lon_c, lat_c, w_naive, n_ev, nD)"""
    vc, lon, lat, n_veh, batt, cum = _demand_context(veh)
    el, ea = low_soc_events(cum, vc, lon, lat, np.ones(n_veh), batt, thr)   # naive 全员满电
    ub_n, w_n, _, _ = aggregate_blocks(el, ea, grid)
    lc, ac = block_centroid(ub_n, grid)
    return dict(master=ub_n, lon_c=lc, lat_c=ac, w_naive=w_n,
                n_ev=int(len(el)), nD=int(len(ub_n)))


# 对初始 SoC 抽样 draws 次的蒙特卡洛需求面。
def monte_carlo_demand_surface(draws=None, veh=None, thr=None, grid=None, seed0=1000):
    """对初始 SoC 蒙特卡洛集成，返回期望需求面。
    return dict(master, lon_c, lat_c, w_exp, appear, reals, n_ev, nD)"""
    draws = draws or C.N_ENSEMBLE
    vc, lon, lat, n_veh, batt, cum = _demand_context(veh)
    reals, n_ev, nD = [], [], []
    for m in range(draws):
        rng = np.random.default_rng(seed0 + m)
        e1, e2 = low_soc_events(cum, vc, lon, lat, tnorm(n_veh, rng), batt, thr)
        ub, w, _, _ = aggregate_blocks(e1, e2, grid)
        reals.append((ub, w)); n_ev.append(len(e1)); nD.append(len(ub))
    master = np.unique(np.concatenate([u for u, _ in reals])) if reals else np.array([], dtype=np.int64)
    lc, ac = block_centroid(master, grid)
    sw = np.zeros(len(master)); appear = np.zeros(len(master))
    for ub, w in reals:
        pos = np.searchsorted(master, ub); sw[pos] += w; appear[pos] += 1
    w_exp = sw / draws
    return dict(master=master, lon_c=lc, lat_c=ac, w_exp=w_exp,
                appear=appear / draws, reals=reals, n_ev=np.array(n_ev), nD=np.array(nD),
                draws=int(draws))

# 将需求面与主栅格对齐，
def _align_to_master(surface, key, master):
    out = np.zeros(len(master), np.float64)
    if surface is None or key not in surface or len(surface["master"]) == 0:
        return out
    pos = np.searchsorted(master, surface["master"])
    out[pos] = surface[key]
    return out


# 把 naive 与蒙特卡洛需求面对齐到同一个 master 栅格。
def align_demand_surfaces(mc=None, naive=None, grid=None):
    """对齐不同需求面的栅格，常用于 naive 与 w_exp 的并列比较。"""
    surfaces = [s for s in (mc, naive) if s is not None and len(s.get("master", [])) > 0]
    master = np.unique(np.concatenate([s["master"] for s in surfaces])) if surfaces else np.array([], dtype=np.int64)
    lc, ac = block_centroid(master, grid)
    out = dict(master=master, lon_c=lc, lat_c=ac)
    out["w_exp"] = _align_to_master(mc, "w_exp", master)
    out["appear"] = _align_to_master(mc, "appear", master)
    out["w_naive"] = _align_to_master(naive, "w_naive", master)
    if mc is not None:
        for k in ("reals", "n_ev", "nD", "draws"):
            if k in mc:
                out[k] = mc[k]
    if naive is not None:
        out["naive_events"] = int(naive.get("n_ev", 0))
        out["naive_nD"] = int(naive.get("nD", 0))
    return out


def demand_surface(draws=None, veh=None, thr=None, grid=None, seed0=1000):
    """兼容旧名称；现在只返回蒙特卡洛期望需求面，不再计算 naive。"""
    return monte_carlo_demand_surface(draws=draws, veh=veh, thr=thr, grid=grid, seed0=seed0)


# ------------------------------- 距离 / 目标 ----------------------------
# 计算需求块到站点的完整距离矩阵 c。
# 输入需求块中心 blon/blat 和站点 slon/slat；输出形状为 (需求块数, 站点数) 的绕行距离矩阵。
def block_station_dist(blon, blat, slon, slat, detour=None, chunk=256):
    detour = C.DETOUR if detour is None else detour
    nD = len(blon); c = np.empty((nD, len(slon)), np.float64)
    for s in range(0, nD, chunk):
        e = min(s + chunk, nD)
        c[s:e] = haversine_km(blon[s:e, None], blat[s:e, None], slon[None, :], slat[None, :]) * detour
    return c


# 目标函数/错配指数 M。输入需求权重 w 和每个需求块到最近站的距离 min_c；覆盖距离内按距离计成本，超过 C_BAR 按抛锚惩罚计成本。
def compute_M(w, min_c, c_bar=None, p_dead=None):
    c_bar = C.C_BAR if c_bar is None else c_bar
    p_dead = C.P_DEAD if p_dead is None else p_dead
    cov = min_c <= c_bar
    return float((w * min_c * cov).sum() + p_dead * (w * (~cov)).sum())


# ---------------------------------- S1: 只增 ----------------------------
# 每轮选择一个让 M 下降最多的候选站；输出候选索引 sel 和每次新增后的 M 历史 M_hist。
def greedy_add(w, min_c_existing, cand_lon, cand_lat, blon, blat, n_max,
               detour=None, c_bar=None, p_dead=None):
    detour = C.DETOUR if detour is None else detour
    cc = haversine_km(blon[:, None], blat[:, None], cand_lon[None, :], cand_lat[None, :]) * detour
    cur = min_c_existing.copy(); M_hist = [compute_M(w, cur, c_bar, p_dead)]
    avail = np.ones(cc.shape[1], bool); sel = []
    for _ in range(n_max):
        nm = np.minimum(cur[:, None], cc)
        red = (w[:, None] * (cur[:, None] - nm)).sum(axis=0); red[~avail] = -1.0
        k = int(red.argmax())
        if red[k] <= 1e-9:
            break
        avail[k] = False; cur = np.minimum(cur, cc[:, k]); sel.append(k)
        M_hist.append(compute_M(w, cur, c_bar, p_dead))
    return sel, M_hist


# ---------------------------------- S2: 只减 / 边际 -------------------------------
# 对每个需求块找最近和次近站；若关闭最近站，成本从 first 变为 second，差额累加到该最近站的 delta。
def marginal_delta(w, c, c_bar=None, p_dead=None):
    c_bar = C.C_BAR if c_bar is None else c_bar
    p_dead = C.P_DEAD if p_dead is None else p_dead
    nD, nF = c.shape
    order = np.argpartition(c, 1, axis=1)[:, :2]
    d0 = c[np.arange(nD), order[:, 0]]; d1 = c[np.arange(nD), order[:, 1]]
    sw = d0 > d1
    nearest = np.where(sw, order[:, 1], order[:, 0])
    first = np.minimum(d0, d1); second = np.maximum(d0, d1)

    # 局部成本函数。距离在覆盖阈值内用实际距离，超过阈值用固定抛锚惩罚。
    def cost(dd):
        return np.where(dd <= c_bar, dd, p_dead)
    delta = np.zeros(nF); np.add.at(delta, nearest, w * (cost(second) - cost(first)))
    return delta, nearest, first, second


# 它把零边际比例拆成 |F|-|D| 造成的结构性下界，以及需求地理聚集造成的真实冗余。
def pigeonhole_decomposition(c, w):
    """把“零边际站占比”拆为结构必然(|F|-|D|)与真实地理聚集。须在与命题相同的需求基上计算。"""
    nD, nF = c.shape
    delta, nearest, _, _ = marginal_delta(w, c)
    n_zero = int((delta == 0).sum()); distinct = int(np.unique(nearest).size); floor = nF - nD
    return dict(nF=nF, nD=nD, n_zero=n_zero, zero_frac=n_zero / nF,
                pigeonhole_floor=floor, floor_frac=floor / nF,
                mechanical_share=floor / max(n_zero, 1), distinct_nearest=distinct)


# 把每个需求块分配给最近站，统计各站承接的需求量。
# 输出 load（每站负载）、g（负载基尼系数）、nloaded（有负载的站点数）。
def event_load(w, c):
    nearest = c.argmin(axis=1); load = np.zeros(c.shape[1]); np.add.at(load, nearest, w)
    x = np.sort(load); nloaded = int((x > 0).sum())
    g = float((2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum())) if x.sum() else 0.0
    return load, g, nloaded


# ------------------------------------- S3: 等量调配 ----------------------------
# 先用 greedy_add 找新增侧初值，再按边际价值移除低价值现有站，最后在候选池中做有限轮 2-opt 改进。
def swap(w, c_existing, cand_lon, cand_lat, blon, blat, n_swap,
         detour=None, c_bar=None, p_dead=None, n_iter_2opt=8):
    """关 n_swap 个最低边际站、开 n_swap 个候选（总数不变），新增侧做向量化 best-improvement 2-opt。"""
    detour = C.DETOUR if detour is None else detour
    c_bar = C.C_BAR if c_bar is None else c_bar
    p_dead = C.P_DEAD if p_dead is None else p_dead
    min_c0 = c_existing.min(axis=1)
    sel, _ = greedy_add(w, min_c0, cand_lon, cand_lat, blon, blat, n_swap, detour, c_bar, p_dead)
    cc = haversine_km(blon[:, None], blat[:, None], cand_lon[None, :], cand_lat[None, :]) * detour
    c_aug = np.concatenate([c_existing, cc[:, sel]], axis=1)
    delta_aug, _, _, _ = marginal_delta(w, c_aug, c_bar, p_dead)
    rm = np.argsort(delta_aug[:c_existing.shape[1]])[:n_swap].tolist()
    keep = np.ones(c_existing.shape[1], bool); keep[rm] = False
    base_min = c_existing[:, keep].min(axis=1)

    # 给定每个需求块当前最近距离 mc，快速计算对应 M。
    def M_min(mc):
        cov = mc <= c_bar
        return float((w * mc * cov).sum() + p_dead * (w * (~cov)).sum())

    # 给定新增候选索引 added，计算“保留站 + 这些新增站”下每个需求块的最近距离。
    def added_min(added):
        return np.minimum(base_min, cc[:, added].min(axis=1)) if added else base_min.copy()

    indiv = (w[:, None] * np.maximum(base_min[:, None] - cc, 0.0)).sum(axis=0)
    poolN = min(cc.shape[1], max(60, 6 * n_swap)); pool = np.argsort(indiv)[::-1][:poolN]
    added = list(sel); M_best = M_min(added_min(added))
    for _ in range(n_iter_2opt):
        improved = False
        for s in range(len(added)):
            others = added[:s] + added[s + 1:]
            om = np.minimum(base_min, cc[:, others].min(axis=1)) if others else base_min
            cand_min = np.minimum(om[:, None], cc[:, pool]); cov = cand_min <= c_bar
            Mk = (w[:, None] * cand_min * cov).sum(0) + p_dead * (w[:, None] * (~cov)).sum(0)
            k = int(np.argmin(Mk))
            if Mk[k] < M_best - 1e-9 and int(pool[k]) not in others:
                added = others + [int(pool[k])]; M_best = float(Mk[k]); improved = True
        if not improved:
            break
    return dict(added=added, removed=rm, M=float(M_best), iters=n_iter_2opt)
