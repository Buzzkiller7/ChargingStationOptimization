#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_cities.py — 批量运行器
================================
遍历 cso_config.CITIES 注册表里**已填好**的每一个城市，复跑 STEP_1~3 的全部图与分析，
并把结果落盘到：  <项目根>/Figure/<city_key>_all_stations/

每个城市目录下生成：
  图（PNG）
    01_trajectory_vs_stations.png      轨迹范围 vs 充电站（全部有效坐标站点）
    02_district_shares.png             二级行政区 GPS 占比 vs 站点占比（需 geopandas+联网，失败自动跳过）
    03_gps_points_per_vehicle.png      每车 GPS 点数分布
    04_speed_distribution.png          瞬时速度分布
    05_soc_traces_grid.png             4 辆代表车的 SoC 轨迹
    06_soc_trace_busiest.png           最繁忙车辆 SoC 轨迹
    07_naive_demand_surface.png        全员满电基线需求面
    08_lowsoc_events_per_draw.png      蒙特卡洛每次抽样低电量事件数
    09_mc_demand_stability.png         需求格出现频率（稳定性）
    10_baseline_demand_vs_stations.png 基线需求 vs 站点（M0）
    11_s1_greedy_add_curve.png         S1 只增：M/M0 曲线
    12_s2_redundancy_load.png          S2 只减：鸽笼分解 + 负载集中度
    13_s2_removal_marginal_audit.png   S2 减站边际效应审查
    14_s3_add_vs_swap.png              S3 等量调配 vs 只增
  分析结果
    metrics.json                       全部数值指标（机器可读）
    summary.txt                        人类可读汇总
    station_supply.csv                 全部站点供给统计（不按建成日/枪数过滤）
    district_distribution.csv          各二级行政区 GPS/站点分布（若可用）
    district_uniformity.csv            分布均衡度（CV/Gini）
    s1_greedy_add.csv                  S1 各新增比例的降幅表
    s2_removal_audit.csv               S2 按低边际站逐批减少后的 M/M0
    s2_top_risk_stations.csv           S2 单站移除风险最高的站点
    s3_swap.csv                        S3 各比例 swap 降幅表

用法
----
    python run_all_cities.py                     # 跑 CITIES 里所有城市
    python run_all_cities.py --cities guangzhou  # 只跑指定城市（逗号分隔）
    python run_all_cities.py --rebuild           # 强制重算缓存（数据或城市变更后用）
    python run_all_cities.py --draws 50          # 调小蒙特卡洛抽样数（更快）
    python run_all_cities.py --fig-root D:/xxx/Figure   # 自定义输出根目录

依赖：numpy pandas matplotlib polars pyarrow；行政区图额外需要 geopandas shapely + 联网。
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

# 让本脚本无论从哪里启动都能 import 同目录的 cso_config / cso_engine
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")  # 无界面后端，直接存图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                         "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

import cso_config as C
import cso_engine as E


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def save(fig, outdir: Path, name: str, log: list) -> None:
    """保存并关闭一张图，记录到 log。"""
    p = outdir / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log.append(f"[fig ] {name}")


def jsonable(obj):
    """把 numpy 标量/数组转成可 JSON 序列化的纯 python。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def _pigeonhole_nonnegative(c, w):
    """兼容旧 engine：鸽笼结构下界按 max(|F|-|D|, 0) 解释。"""
    pig = dict(E.pigeonhole_decomposition(c, w))
    nF = int(pig["nF"])
    n_zero = int(pig["n_zero"])
    floor = max(int(pig.get("pigeonhole_floor", 0)), 0)
    pig["pigeonhole_floor"] = floor
    pig["floor_frac"] = floor / max(nF, 1)
    pig["mechanical_share"] = floor / max(n_zero, 1)
    return pig


def _s2_removal_audit(w, c, delta, load, slon, slat, nF, add_counts, M0, outdir: Path):
    """按 STEP_3 的 S2 审查口径，计算逐批减站后的真实 M/M0。"""
    eps = max(1e-9, 1e-10 * max(M0, 1.0))
    if M0 <= eps:
        raise ValueError("M0 过小，不能稳定计算 S2 减站边际效应。")

    order_low = np.argsort(delta)
    zero_n = int((delta <= eps).sum())
    pos = delta[delta > eps]
    q = {}
    if len(pos):
        for name, value in zip(["p50", "p90", "p99", "max"],
                               np.percentile(pos / M0, [50, 90, 99, 100])):
            q[name] = float(100 * value)

    scenario_counts = np.asarray(add_counts, dtype=int)
    max_remove = min(nF - 1, max(int(0.30 * nF),
                                 zero_n + int(0.10 * nF),
                                 int(scenario_counts.max())))
    grid_counts = np.linspace(0, max_remove, 61).astype(int)
    remove_counts = np.unique(np.r_[grid_counts, scenario_counts,
                                    zero_n, min(nF - 1, zero_n + 1)])
    remove_counts = remove_counts[(remove_counts >= 0) & (remove_counts < nF)]

    M_removed = []
    for k in remove_counts:
        if k == 0:
            mc = c.min(axis=1)
        else:
            keep = np.ones(nF, dtype=bool)
            keep[order_low[:k]] = False
            mc = c[:, keep].min(axis=1)
        M_removed.append(E.compute_M(w, mc, c_bar=C.C_BAR, p_dead=C.P_DEAD))
    M_removed = np.asarray(M_removed)
    curve = pd.DataFrame({
        "removed_n": remove_counts,
        "removed_pct": 100 * remove_counts / nF,
        "M": M_removed,
        "M_over_M0": M_removed / M0,
        "M_increase_pct": 100 * (M_removed / M0 - 1),
    })

    key_counts = np.unique(np.r_[0, scenario_counts, zero_n, min(nF - 1, zero_n + 1)])
    summary = curve[curve["removed_n"].isin(key_counts)].copy()
    summary["scenario"] = "检查点"
    summary.loc[summary["removed_n"] == 0, "scenario"] = "不减站"
    if zero_n > 0:
        summary.loc[summary["removed_n"] == zero_n, "scenario"] = "移除全部初始零边际站"
    if zero_n + 1 < nF:
        summary.loc[summary["removed_n"] == min(nF - 1, zero_n + 1), "scenario"] = "零边际后再减1站"
    for frac, cnt in zip(C.ADD_STATION_FRACS, scenario_counts):
        summary.loc[summary["removed_n"] == cnt, "scenario"] = "减%d%%站" % int(frac * 100)

    summary_show = summary.rename(columns={
        "scenario": "情景",
        "removed_n": "减站数",
        "removed_pct": "减站比例%",
        "M": "M",
        "M_over_M0": "M/M0",
        "M_increase_pct": "M增幅%",
    })
    summary_show[["情景", "减站数", "减站比例%", "M", "M/M0", "M增幅%"]].to_csv(
        outdir / "s2_removal_audit.csv", index=False, encoding="utf-8-sig")

    risk = pd.DataFrame({
        "station": np.arange(nF),
        "lon": slon,
        "lat": slat,
        "delta_M": delta,
        "delta_M_over_M0_pct": 100 * delta / M0,
        "assigned_load": load,
    }).sort_values("delta_M", ascending=False).head(10)
    risk_show = risk.rename(columns={
        "station": "站点序号",
        "lon": "经度",
        "lat": "纬度",
        "delta_M": "单站ΔM",
        "delta_M_over_M0_pct": "单站ΔM/M0%",
        "assigned_load": "最近分配负载",
    })
    risk_show.to_csv(outdir / "s2_top_risk_stations.csv", index=False, encoding="utf-8-sig")

    return curve, summary_show, risk, {
        "zero_near_zero_n": zero_n,
        "zero_near_zero_frac": float(zero_n / nF),
        "positive_n": int((delta > eps).sum()),
        "positive_delta_quantiles_pct": q,
    }


# --------------------------------------------------------------------------- #
# STEP 1 · 数据探索
# --------------------------------------------------------------------------- #
def step1(cfg, outdir: Path, metrics: dict, log: list):
    b = cfg["bbox"]
    sc = cfg["station_cols"]
    pl = E.pl

    lf = pl.scan_parquet(str(cfg["raw_path"]))
    rows = lf.select(pl.len()).collect().item()
    nveh = lf.select(pl.col("vehicle_id").n_unique()).collect().item()
    metrics["step1"] = {"gps_records": int(rows), "vehicles": int(nveh)}
    log.append(f"[s1  ] GPS 记录 {rows:,} | 车辆 {nveh:,}")

    # ---- 1.2 站点供给：全部站点口径（不按建成日/枪数过滤） ----
    raw = pd.read_csv(cfg["stations_path"])
    lon = pd.to_numeric(raw[sc["lon"]], errors="coerce")
    lat = pd.to_numeric(raw[sc["lat"]], errors="coerce")
    has_coord = lon.notna() & lat.notna()
    st = raw.loc[has_coord].copy()
    st["_lon"] = lon.loc[has_coord]
    st["_lat"] = lat.loc[has_coord]
    in_bbox = (st["_lon"].between(b["lon_min"], b["lon_max"]) &
               st["_lat"].between(b["lat_min"], b["lat_max"]))
    out_of_bbox = int((~in_bbox).sum())
    st = st.loc[in_bbox].copy()
    fast = pd.to_numeric(st[sc["fast"]], errors="coerce").fillna(0)
    slow = (pd.to_numeric(st[sc["slow"]], errors="coerce").fillna(0)
            if sc.get("slow") in st else pd.Series(np.zeros(len(st)), index=st.index))
    st["lon"] = st["_lon"].astype(float)
    st["lat"] = st["_lat"].astype(float)
    st["fast"] = fast.astype(float)
    st["slow"] = slow.astype(float)
    fast_k = int(st["fast"].sum())
    slow_k = int(st["slow"].sum())
    zero_gun = int(((st["fast"] + st["slow"]) <= 0).sum())
    supply = dict(
        policy="all_stations_no_date_or_gun_filter",
        raw=int(len(raw)),
        used=int(len(st)),
        missing_coord=int((~has_coord).sum()),
        out_of_bbox=out_of_bbox,
        fast_guns=fast_k,
        slow_guns=slow_k,
        total_guns=fast_k + slow_k,
        zero_gun_stations=zero_gun,
    )
    metrics["station_supply"] = jsonable(supply)
    pd.DataFrame([{
        "城市": cfg["label"],
        "站点口径": "全部站点：不按建成日过滤，不按枪数>0过滤",
        "原始站点表": supply["raw"],
        "实际使用站点": supply["used"],
        "缺坐标剔除": supply["missing_coord"],
        "越界剔除": supply["out_of_bbox"],
        "零枪站点数（保留）": supply["zero_gun_stations"],
        "快充枪": supply["fast_guns"],
        "慢充枪": supply["slow_guns"],
        "合计枪数": supply["total_guns"],
    }]).to_csv(outdir / "station_supply.csv", index=False, encoding="utf-8-sig")
    log.append(f"[s1  ] 全部站点口径 {supply['raw']} -> {supply['used']} "
               f"(缺坐标 {supply['missing_coord']} | 越界 {supply['out_of_bbox']} | "
               f"零枪保留 {supply['zero_gun_stations']})")

    # ---- 1.3 轨迹范围 vs 站点（画全部有效坐标站点） ----
    n_sample = min(rows, 80000)
    samp = (lf.select(["lon", "lat"])
            .filter(pl.col("lon").is_between(b["lon_min"], b["lon_max"]) &
                    pl.col("lat").is_between(b["lat_min"], b["lat_max"]))
            .gather_every(max(1, rows // n_sample)).collect())
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    E.draw_city_context(ax=ax, cfg=cfg, show_admin=True, show_grid=True,
                        grid_alpha=0.18, grid_linewidth=0.25, warn=False)
    ax.scatter(samp["lon"], samp["lat"], s=1, c="#000000", alpha=0.5,
               label="GPS 抽样点", zorder=4)
    ax.scatter(st["lon"], st["lat"], s=2, c="#ff0000", alpha=0.6, marker="x",
               label="充电站（全部口径）", zorder=5)
    ax.set_title("轨迹范围 vs 充电站  |  %s" % cfg["label"])
    ax.legend(loc="lower right")
    save(fig, outdir, "01_trajectory_vs_stations.png", log)

    # ---- 1.3b 行政区分布（可选：geopandas + 联网） ----
    try:
        dd = E.district_distribution(lf, rows=rows, cfg=cfg)
        dist = dd["dist"]
        dist.to_csv(outdir / "district_distribution.csv", index=False, encoding="utf-8-sig")
        dd["uniformity"].to_csv(outdir / "district_uniformity.csv", index=False, encoding="utf-8-sig")
        metrics["step1"]["n_districts"] = int(len(dist))
        metrics["step1"]["uniformity"] = jsonable(dd["uniformity"].to_dict("list"))
        metrics["step1"]["district_meta"] = jsonable({
            "city_name": dd.get("city_name"),
            "city_adcode": dd.get("city_adcode"),
            "admin_col": dd.get("admin_col"),
            "boundary_note": dd.get("boundary_note"),
        })
        log.append("[s1  ] 行政区分布 %d 个二级分区 | %s" %
                   (len(dist), dd.get("boundary_note", "")))

        map_gdf = dd["map_gdf"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        for ax, (col, title) in zip(axes, [("gps_share", "GPS 记录占比"),
                                           ("station_share", "充电站占比")]):
            map_gdf.plot(column=col, cmap="Blues", linewidth=0.8, edgecolor="white",
                         legend=True, ax=ax,
                         missing_kwds={"color": "#f0f0f0", "label": "无数据"})
            E.draw_city_context(ax=ax, cfg=cfg, admin_gdf=map_gdf, show_admin=True,
                                show_grid=True, show_labels=True, boundary_linewidth=1.0,
                                grid_linewidth=0.3, grid_alpha=0.4, xlabel=None,
                                ylabel=None, warn=False)
            ax.set_title(title)
            ax.set_axis_off()
        dd["station_gdf"].plot(ax=axes[1], color="#851B08", markersize=0.5, alpha=0.5)
        plt.suptitle(f"{cfg['label']} 二级行政区分布对比", y=0.98)
        save(fig, outdir, "02_district_shares.png", log)
    except Exception as exc:
        log.append(f"[s1  ] 行政区分布跳过：{type(exc).__name__}: {exc}")
        metrics["step1"]["district_error"] = f"{type(exc).__name__}: {exc}"

    # ---- 1.4 每车 GPS 点数分布 ----
    cnt = lf.group_by("vehicle_id").agg(pl.len().alias("n")).collect()["n"].to_numpy()
    metrics["step1"]["pts_per_vehicle"] = {
        "median": float(np.median(cnt)), "mean": float(cnt.mean()), "max": int(cnt.max())}
    fig = plt.figure(figsize=(7, 3))
    plt.hist(cnt[cnt < np.percentile(cnt, 99)], bins=60, color="#2b6cb0")
    plt.title("GPS points per vehicle (<99th pct)")
    plt.xlabel("points"); plt.ylabel("vehicles")
    save(fig, outdir, "03_gps_points_per_vehicle.png", log)

    # ---- 1.5 速度分布 ----
    spd = lf.select("speed_kmh").collect()["speed_kmh"].to_numpy()
    spd = spd[(spd > 0) & (spd < 120)]
    metrics["step1"]["speed_kmh"] = {"median": float(np.median(spd)), "mean": float(spd.mean())}
    fig = plt.figure(figsize=(7, 3))
    plt.hist(spd, bins=60, color="#dd6b20")
    plt.title("Instantaneous speed (0-120 km/h)")
    plt.xlabel("km/h"); plt.ylabel("count")
    save(fig, outdir, "04_speed_distribution.png", log)


# --------------------------------------------------------------------------- #
# STEP 2 · SoC → 需求面
# --------------------------------------------------------------------------- #
def step2(cfg, outdir: Path, metrics: dict, log: list, draws: int):
    z = E.load_segments()
    vc, lon, lat = z["vc"], z["lon"], z["lat"]
    d, dt, v, start = z["d"], z["dt"], z["v"], z["start"]
    n_veh = int(z["n_veh"])
    batt = C.VEHICLES[C.VEH_DEFAULT]["batt"]
    cum = E.cumulative_kwh(d, dt, v, vc, start, C.VEH_DEFAULT)
    med_kwh = float(np.median(cum[~start]))
    metrics["step2"] = {"veh_default": C.VEH_DEFAULT, "batt_kwh": batt,
                        "median_cum_kwh": med_kwh}
    log.append(f"[s2  ] 单车累计耗电中位 {med_kwh:.1f} kWh")

    # ---- 2.1 代表车 SoC 轨迹（按记录数百分位取 4 辆） ----
    counts = np.bincount(vc, minlength=n_veh)
    veh_pool = np.flatnonzero(counts > 0)
    veh_sorted = veh_pool[np.argsort(counts[veh_pool])]
    ps = [0.3, 0.5, 0.7, 0.9]
    veh_show = [int(veh_sorted[min(int(round((len(veh_sorted) - 1) * p)),
                                   len(veh_sorted) - 1)]) for p in ps]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for ax, veh in zip(axes.ravel(), veh_show):
        idx = np.where(vc == veh)[0]
        soc_v = 1 - (cum[idx] % batt) / batt
        ax.plot(soc_v, lw=1, color="#2b6cb0")
        ax.axhline(C.SOC_LOW, ls="--", color="#c53030", lw=1)
        ax.set_title("veh #%d | records=%d" % (veh, len(idx)))
        ax.set_xlabel("GPS point index"); ax.set_ylabel("SoC")
        ax.set_ylim(-0.05, 1.05); ax.set_xlim(left=-1, right=len(soc_v))
    plt.tight_layout(h_pad=1.5)
    save(fig, outdir, "05_soc_traces_grid.png", log)

    # ---- 2.1b 最繁忙车辆 ----
    veh0 = int(np.argsort(np.bincount(vc))[-2])
    idx0 = np.where(vc == veh0)[0]
    soc = 1 - (cum[idx0] - np.floor(cum[idx0] / batt) * batt) / batt
    fig = plt.figure(figsize=(10, 3))
    plt.plot(soc, lw=1, color="#2b6cb0")
    plt.axhline(C.SOC_LOW, ls="--", color="#c53030",
                label="low-SoC threshold %.0f%%" % (100 * C.SOC_LOW))
    plt.title("SoC trace of busiest vehicle #%d (start full)" % veh0)
    plt.xlabel("GPS point"); plt.ylabel("SoC"); plt.legend()
    save(fig, outdir, "06_soc_trace_busiest.png", log)

    # ---- 2.2 naive 需求面 ----
    D_naive = E.naive_demand_surface()
    metrics["step2"]["naive_events"] = int(D_naive["n_ev"])
    metrics["step2"]["naive_nD"] = int(D_naive["nD"])
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    E.draw_city_context(ax=ax, cfg=cfg, show_admin=True, show_grid=True,
                        grid_alpha=0.25, grid_linewidth=0.35, warn=False)
    s = ax.scatter(D_naive["lon_c"], D_naive["lat_c"], c=D_naive["w_naive"],
                   cmap="OrRd", s=5, zorder=4)
    plt.colorbar(s, ax=ax, label="events / cell", fraction=0.046)
    ax.set_title("naive demand (%d events, %d cells)" % (D_naive["n_ev"], D_naive["nD"]))
    save(fig, outdir, "07_naive_demand_surface.png", log)

    # ---- 2.3 蒙特卡洛集成 ----
    t0 = time.time()
    D_mc = E.monte_carlo_demand_surface(draws=draws)
    D = E.align_demand_surfaces(D_mc, D_naive)
    metrics["step2"].update({
        "draws": int(draws),
        "mc_events_mean": float(D["n_ev"].mean()), "mc_events_std": float(D["n_ev"].std()),
        "mc_nD_mean": float(D["nD"].mean()), "mc_nD_std": float(D["nD"].std()),
        "master_cells": int(len(D["master"])),
        "ratio_vs_naive": float(D["n_ev"].mean() / max(D_naive["n_ev"], 1)),
        "mc_time_sec": round(time.time() - t0, 1),
    })
    log.append("[s2  ] MC %d 次: 事件 %.0f±%.0f | 需求块 %.0f±%.0f | 主网格 %d"
               % (draws, D["n_ev"].mean(), D["n_ev"].std(),
                  D["nD"].mean(), D["nD"].std(), len(D["master"])))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(D["n_ev"], bins=10, color="#2b6cb0")
    ax.axvline(D_naive["n_ev"], ls="--", lw=2, color="#c53030",
               label="naive=%d" % D_naive["n_ev"])
    ax.set_title("low-SoC events per draw")
    ax.set_xlabel("events"); ax.set_ylabel("draws"); ax.legend()
    save(fig, outdir, "08_lowsoc_events_per_draw.png", log)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    E.draw_city_context(ax=ax, cfg=cfg, show_admin=True, show_grid=True,
                        grid_alpha=0.20, grid_linewidth=0.25, warn=False)
    m = D_mc["w_exp"] > 0
    s2 = ax.scatter(D_mc["lon_c"][m], D_mc["lat_c"][m], c=D_mc["appear"][m],
                    cmap="viridis", s=10, vmin=0, vmax=1, zorder=4)
    plt.colorbar(s2, ax=ax, label="appearance frequency across MC draws", fraction=0.046)
    ax.set_title("Monte Carlo demand-cell stability (%d mean events, %d cells)"
                 % (round(D_mc["n_ev"].mean()), len(D_mc["master"])))
    save(fig, outdir, "09_mc_demand_stability.png", log)

    return D  # 给 STEP_3 复用


# --------------------------------------------------------------------------- #
# STEP 3 · 基线 M0 与 S1/S2/S3
# --------------------------------------------------------------------------- #
def step3(cfg, outdir: Path, metrics: dict, log: list, D):
    st = E.load_stations()
    slon = st["lon"].astype(float)
    slat = st["lat"].astype(float)
    nF = len(slon)
    lc, ac, w_exp, w_naive = D["lon_c"], D["lat_c"], D["w_exp"], D["w_naive"]
    c = E.block_station_dist(lc, ac, slon, slat)
    min_c = c.min(axis=1)

    def stats(w):
        M0 = E.compute_M(w, min_c, c_bar=C.C_BAR, p_dead=C.P_DEAD)
        dead = float(((min_c > C.C_BAR) * w).sum() / max(w.sum(), 1e-12))
        return M0, M0 / max(w.sum(), 1e-12), float((w * min_c).sum() / max(w.sum(), 1e-12)), dead

    Me, me, ne, de = stats(w_exp)
    Mn, mn, nn, dn = stats(w_naive)
    metrics["step3"] = {
        "nF_stations_used": int(nF), "n_demand_cells": int(len(lc)),
        "C_BAR": C.C_BAR, "P_DEAD": C.P_DEAD,
        "expected": {"M0": Me, "mean_cost": me, "mean_nearest_km": ne, "dead_rate": de},
        "naive": {"M0": Mn, "mean_cost": mn, "mean_nearest_km": nn, "dead_rate": dn},
    }
    log.append("[s3  ] 站点 %d | 需求格 %d | M0(exp)=%.0f 抛锚率=%.4f" % (nF, len(lc), Me, de))

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    m = w_exp > 0
    ax.scatter(slon, slat, s=2, c="#2b6cb0", alpha=0.4, marker="x",
               label="stations", zorder=5)
    E.draw_city_context(ax=ax, cfg=cfg, show_admin=True, show_grid=True,
                        grid_alpha=0.20, grid_linewidth=0.25, warn=False)
    s = ax.scatter(lc[m], ac[m], c=w_exp[m], cmap="OrRd", s=12,
                   label="expected demand", zorder=4)
    plt.colorbar(s, ax=ax, label="exp. events/cell", fraction=0.046)
    ax.set_title("Baseline: demand vs stations  M0=%.0f" % Me)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, outdir, "10_baseline_demand_vs_stations.png", log)

    # ---- S1 只增 ----
    add_counts = C.add_station_counts(nF)
    add_labels = [f"{int(frac * 100)}%" for frac in C.ADD_STATION_FRACS]
    cand = np.argsort(w_exp)[::-1][:min(C.CAND_CAP, len(w_exp))]
    sel, hist = E.greedy_add(w_exp, min_c, lc[cand], ac[cand], lc, ac, max(add_counts))
    ratio = np.array(hist) / Me
    rat = lambda n: ratio[n] if len(ratio) > n else ratio[-1]
    s1_tbl = pd.DataFrame({
        "新增比例": add_labels, "新增站数": list(add_counts),
        "M/M0": [round(rat(n), 4) for n in add_counts],
        "降幅%": [round(100 * (1 - rat(n)), 1) for n in add_counts],
    })
    s1_tbl.to_csv(outdir / "s1_greedy_add.csv", index=False, encoding="utf-8-sig")
    metrics["step3"]["s1_greedy_add"] = s1_tbl.to_dict("records")
    fig = plt.figure(figsize=(6.4, 3.6))
    plt.plot(range(len(ratio)), ratio, "-o", ms=3, color="#2b6cb0")
    plt.scatter(list(add_counts), [rat(n) for n in add_counts], color="#c53030", zorder=5)
    plt.title("S1 greedy add: M/M0"); plt.xlabel("stations added N"); plt.ylabel("M/M0")
    save(fig, outdir, "11_s1_greedy_add_curve.png", log)

    # ---- S2 只减 / 边际 ----
    mN = w_naive > 0
    pig = _pigeonhole_nonnegative(c[mN], w_naive[mN])
    mE = w_exp > 0
    floorE = _pigeonhole_nonnegative(c[mE], w_exp[mE])["floor_frac"]
    delta, nearest, _, _ = E.marginal_delta(w_exp, c)
    load, gini, nl = E.event_load(w_exp, c)
    ss = np.sort(delta)[::-1]
    top_count = max(1, int(np.ceil(0.1 * nF)))
    cumv = np.cumsum(ss) / ss.sum() if ss.sum() > 0 else np.zeros_like(ss)
    top10 = float(cumv[min(top_count - 1, len(cumv) - 1)]) if len(cumv) else 0.0
    metrics["step3"]["s2"] = jsonable({
        "pigeonhole_naive": pig, "floor_frac_expected": floorE,
        "top10pct_marginal_share": top10, "load_gini": gini, "n_loaded_stations": nl,
    })
    obs = 100 * pig["zero_frac"]; fl = 100 * pig["floor_frac"]; gen = obs - fl
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bars = ax[0].bar(["观测 Δ=0", "鸽笼下界", "真实聚集"],
                     [obs, fl, gen], color=["#718096", "#a0aec0", "#2f855a"])
    for bbar, vv in zip(bars, [obs, fl, gen]):
        ax[0].text(bbar.get_x() + bbar.get_width() / 2, vv + 1, "%.1f%%" % vv,
                   ha="center", fontweight="bold")
    ax[0].set_ylabel("% of stations")
    ax[0].set_title("S2 零边际分解（naive）")
    ld = np.sort(load)[::-1]
    cl = np.cumsum(ld) / ld.sum() if ld.sum() > 0 else np.zeros_like(ld)
    ax[1].plot(np.arange(1, len(ld) + 1) / len(ld), cl, color="#2b6cb0")
    ax[1].plot([0, 1], [0, 1], "--", color="#a0aec0")
    ax[1].set_title("负载集中度（Gini=%.2f）" % gini)
    ax[1].set_xlabel("站点累计比例"); ax[1].set_ylabel("需求累计占比")
    save(fig, outdir, "12_s2_redundancy_load.png", log)

    # ---- S2 减站边际效应审查 ----
    curve, s2_summary, s2_risk, audit = _s2_removal_audit(
        w_exp, c, delta, load, slon, slat, nF, add_counts, Me, outdir)
    metrics["step3"]["s2"].update(jsonable({
        "removal_audit": audit,
        "removal_summary": s2_summary.to_dict("records"),
        "top_risk_stations": s2_risk.to_dict("records"),
    }))

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    eps = max(1e-9, 1e-10 * max(Me, 1.0))
    order_low = np.argsort(delta)
    zero_n = audit["zero_near_zero_n"]
    rank_frac = np.arange(1, nF + 1) / nF
    d_low = delta[order_low]
    ax[0].plot(rank_frac, 100 * d_low / Me, color="#2b6cb0", lw=1.7)
    ax[0].axvline(zero_n / nF, color="#c53030", ls="--", lw=1.2)
    y0, y1 = ax[0].get_ylim()
    ax[0].text(min(zero_n / nF + 0.01, 0.95), y0 + 0.82 * (y1 - y0),
               "零/近零边际\n%.1f%%" % (100 * zero_n / nF),
               color="#c53030", fontsize=8)
    ax[0].set_title("S2 单站移除边际损失")
    ax[0].set_xlabel("站点按 ΔM 从低到高排序")
    ax[0].set_ylabel("单站移除 ΔM / M0 (%)")

    ax[1].plot(curve["removed_pct"], curve["M_over_M0"], "-o", ms=3, color="#2f855a")
    ax[1].axhline(1.0, color="#a0aec0", ls="--", lw=1)
    for frac, cnt in zip(C.ADD_STATION_FRACS, add_counts):
        row = curve[curve["removed_n"] == cnt]
        if len(row):
            x0 = float(row["removed_pct"].iloc[0]); y0 = float(row["M_over_M0"].iloc[0])
            ax[1].scatter([x0], [y0], color="#c53030", zorder=5)
            ax[1].text(x0, y0, " 减%d%%" % int(frac * 100), fontsize=8, va="bottom")
    ax[1].set_title("按低 ΔM 减站后重算 M")
    ax[1].set_xlabel("累计减站比例 (%)")
    ax[1].set_ylabel("M / M0")

    pos_sum = delta.sum()
    if pos_sum > eps:
        d_desc = np.sort(delta)[::-1]
        cum_delta = np.cumsum(d_desc) / pos_sum
        ax[2].plot(rank_frac, cum_delta, color="#805ad5", lw=1.8)
        ax[2].plot([0, 1], [0, 1], "--", color="#a0aec0", lw=1)
        ax[2].set_ylim(0, 1.02)
    else:
        ax[2].text(0.5, 0.5, "所有 ΔM 近似为 0", ha="center", va="center")
    ax[2].set_title("移除风险集中度")
    ax[2].set_xlabel("高风险站点累计比例")
    ax[2].set_ylabel("累计 ΔM 占比")
    plt.tight_layout()
    save(fig, outdir, "13_s2_removal_marginal_audit.png", log)
    log.append("[s3  ] S2 减站审查: 零/近零边际 %d/%d，减站曲线已重算 M" % (zero_n, nF))

    # ---- S3 等量调配 ----
    s3r = {}
    for label, N in zip(add_labels, add_counts):
        r = E.swap(w_exp, c, lc[cand], ac[cand], lc, ac, N)
        s3r[N] = r["M"] / Me
    s3_tbl = pd.DataFrame({
        "新增比例": add_labels, "调配站数": list(add_counts),
        "S1降幅%": [round(100 * (1 - rat(N)), 1) for N in add_counts],
        "S3降幅%": [round(100 * (1 - s3r[N]), 1) for N in add_counts],
    })
    s3_tbl.to_csv(outdir / "s3_swap.csv", index=False, encoding="utf-8-sig")
    metrics["step3"]["s3_swap"] = s3_tbl.to_dict("records")
    x = np.arange(len(add_counts)); ww = 0.38
    fig = plt.figure(figsize=(6.2, 3.6))
    plt.bar(x - ww / 2, [100 * (1 - rat(N)) for N in add_counts], ww,
            label="S1 add", color="#2b6cb0")
    plt.bar(x + ww / 2, [100 * (1 - s3r[N]) for N in add_counts], ww,
            label="S3 swap", color="#2f855a")
    plt.xticks(x, [f"{lab}\nN={N}" for lab, N in zip(add_labels, add_counts)])
    plt.ylabel("reduction %"); plt.title("Add vs revenue-neutral swap"); plt.legend()
    save(fig, outdir, "14_s3_add_vs_swap.png", log)


# --------------------------------------------------------------------------- #
# 汇总写盘
# --------------------------------------------------------------------------- #
def write_summary(cfg, outdir: Path, metrics: dict, log: list):
    (outdir / "metrics.json").write_text(
        json.dumps(jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    supply = metrics.get("station_supply", {})
    s3 = metrics.get("step3", {})
    s2 = s3.get("s2", {})
    audit = s2.get("removal_audit", {})
    rem_rows = s2.get("removal_summary", [])
    rem_brief = []
    for row in rem_rows:
        if row.get("情景") in {"减1%站", "减2%站", "减5%站", "减10%站", "移除全部初始零边际站"}:
            rem_brief.append(
                f"  {row.get('情景')}: 减站 {row.get('减站数')}, "
                f"M/M0={float(row.get('M/M0', 0)):.4f}, "
                f"M增幅={float(row.get('M增幅%', 0)):.2f}%"
            )
    lines = [
        f"城市: {cfg['label']}  (key={cfg['key']})",
        f"轨迹文件: {Path(cfg['raw']).name}",
        "=" * 60,
        "【站点供给——全部站点口径】",
        "  不按建成日过滤，也不按充电枪数量>0过滤；仅剔除缺坐标或越出城市 bbox 的记录。",
        f"  原始站点表: {supply.get('raw')} 行",
        f"  实际用于优化/绘图: {supply.get('used')} 站",
        f"    缺坐标: {supply.get('missing_coord')} | 越界: {supply.get('out_of_bbox')}",
        f"    零枪站: {supply.get('zero_gun_stations')}（保留）",
        f"    快充枪: {supply.get('fast_guns')} | 慢充枪: {supply.get('slow_guns')} "
        f"| 合计: {supply.get('total_guns')}",
        "",
        "【需求与基线】",
        f"  蒙特卡洛事件均值: {metrics.get('step2', {}).get('mc_events_mean')}",
        f"  基线 M0 (期望需求): {s3.get('expected', {}).get('M0')}",
        f"  抛锚率: {s3.get('expected', {}).get('dead_rate')}",
        f"  最近站均距(km): {s3.get('expected', {}).get('mean_nearest_km')}",
        "",
        "【S2 减站边际效应审查】",
        f"  单站零/近零边际: {audit.get('zero_near_zero_n')} "
        f"({100 * audit.get('zero_near_zero_frac', 0):.1f}%)",
        f"  正边际站: {audit.get('positive_n')}",
        "  逐批减站 M/M0:",
        *(rem_brief or ["  （未生成）"]),
        "",
        "【运行日志】",
    ] + ["  " + x for x in log]
    (outdir / "summary.txt").write_text("\n".join(str(x) for x in lines), encoding="utf-8")


def run_city(key: str, fig_root: Path, draws: int, rebuild: bool) -> dict:
    cfg = C.use_city(key)
    tag = cfg.get("trace_date") or "all_stations"
    outdir = fig_root / f"{key}_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {"city": key, "label": cfg["label"], "station_policy": "all_stations",
               "raw": cfg["raw"], "stations": cfg["stations"]}
    log: list = []
    print(f"\n=== {cfg['label']}  ->  {outdir}")

    # 缓存：数据或城市变更、或 --rebuild 时强制重算（precompute 同时重建分段+站点缓存）
    if rebuild or not C.cache_file("segments").exists() or not C.cache_file("stations").exists():
        print("  预计算缓存 ...")
        pc = E.precompute_segments()
        log.append(f"[init] precompute: {pc}")
        print("   ", pc)
    else:
        print("  复用已有缓存（本版本站点缓存为全部站点口径）。")

    try:
        step1(cfg, outdir, metrics, log)
    except Exception as exc:
        log.append(f"[ERR ] STEP_1 失败: {type(exc).__name__}: {exc}")
        print("  STEP_1 失败:", exc)
        traceback.print_exc()

    try:
        D = step2(cfg, outdir, metrics, log, draws)
        step3(cfg, outdir, metrics, log, D)
    except Exception as exc:
        log.append(f"[ERR ] STEP_2/3 失败: {type(exc).__name__}: {exc}")
        print("  STEP_2/3 失败:", exc)
        traceback.print_exc()

    write_summary(cfg, outdir, metrics, log)
    print(f"  完成 -> {outdir}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="批量运行 STEP_1~3，按城市+日期归档图与分析。")
    ap.add_argument("--cities", default="", help="逗号分隔的城市 key；默认 CITIES 里全部")
    ap.add_argument("--fig-root", default="", help="输出根目录；默认 <项目根>/Figure")
    ap.add_argument("--draws", type=int, default=C.N_ENSEMBLE, help="蒙特卡洛抽样数")
    ap.add_argument("--rebuild", action="store_true", help="强制重算缓存")
    args = ap.parse_args()

    cities = [s.strip() for s in args.cities.split(",") if s.strip()] or list(C.CITIES)
    unknown = [k for k in cities if k not in C.CITIES]
    if unknown:
        raise SystemExit(f"未注册的城市: {unknown}。可选: {list(C.CITIES)}")

    fig_root = Path(args.fig_root) if args.fig_root else (C.DATA_ROOT / "Figure")
    fig_root.mkdir(parents=True, exist_ok=True)
    print(f"输出根目录: {fig_root}\n城市: {cities} | draws={args.draws} | rebuild={args.rebuild}")

    all_metrics = {}
    for key in cities:
        try:
            all_metrics[key] = run_city(key, fig_root, args.draws, args.rebuild)
        except Exception as exc:
            print(f"!! 城市 {key} 整体失败: {exc}")
            traceback.print_exc()
            all_metrics[key] = {"error": f"{type(exc).__name__}: {exc}"}

    (fig_root / "ALL_cities_metrics.json").write_text(
        json.dumps(jsonable(all_metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部完成。汇总: {fig_root / 'ALL_cities_metrics.json'}")


if __name__ == "__main__":
    main()
