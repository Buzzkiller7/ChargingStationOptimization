# -*- coding: utf-8 -*-
"""
run_all.py — 年度版批处理入口（第七阶段重写）。

研究框架：用**固定的低电量需求面**（由现有轨迹年生成，当前广州=2019），配合**逐年充电站存量**
（station_snapshot(Y)：create_time <= Y-12-31），逐年做现状错配 baseline + S1/S2/S3 策略 + 与 Y+1
真实新增对照。**每个年份单独出成果**：各自的子目录、各自的曲线/地图/CSV，绝不把不同年份混在同一张图里。
跨年只产出一张汇总表(yearly_summary.csv)和一张逐年分项柱状图（每年一根柱，非叠线）。

防时间泄漏：年份 Y 的推荐只用截至 Y 年底的存量与固定需求面；真实新增(Y→Y+1)只读、不参与候选/打分；
最后一年无 Y+1 时只出推荐、不做对照。

用法：
    python run_all.py                       # 全年份、S1/S2/S3 全跑
    python run_all.py --years 2019,2021,2023
    python run_all.py --strategies s1       # 只跑 S1（逗号分隔 s1,s2,s3 任意子集）
    python run_all.py --backend fast        # CELF 后端 fast(默认 dense，省内存适合全比例)
    SMOKE=1 python run_all.py               # 冒烟：少抽样、少候选，写到 Outputs/_smoke 不覆盖正式结果
输出根：Outputs/<city>/yearly/<Y>/   +   Outputs/<city>/yearly_summary.csv

说明：需求一律映射到路网节点中心、距离一律 OSMnx 路网最短路；需求可视化用路段着色(draw_network_demand)，
不画需求散点；图件只存 PNG。所有比例/参数集中在 cso.py 顶部，本文件不写死城市/年份/比例。
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cso
import style

style.set_nature()
SMOKE = os.environ.get("SMOKE", "").strip().lower() in {"1", "true", "yes"}
SCEN_CN = {"s1": "S1 只增", "s2": "S2 只减", "s3": "S3 等量调配"}
SCEN_SIGN = {"s1": "+", "s2": "-", "s3": "±"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _save(fig, path):
    """统一存 PNG，并关闭释放内存。"""
    style.savefig(fig, path)
    plt.close(fig)


def _demand_norm(w):
    m = w > 0
    vmax = float(np.percentile(w[m], 98)) if np.any(m) else 1.0
    return mcolors.PowerNorm(gamma=0.72, vmin=0.0, vmax=max(vmax, 1e-9))


def curve_rows(scen_cn, r, counts, fracs):
    """情景降幅曲线逐步表：每步一行，含总 M 与三项分解(access/crowd/reach)及各自相对基线比值，
    并在各比例节点标注比例标签。r 为 run_s1_add/run_s2_remove/run_s3_swap 的结果 dict。"""
    M = np.asarray(r["M"], float)
    base = M[0] if M.size and M[0] else np.nan
    comp = {k: np.asarray(r[k], float) for k in ("M_access", "M_crowd", "M_reach")
            if k in r and len(r[k]) == len(M)}
    cbase = {k: (v[0] if v.size and v[0] else np.nan) for k, v in comp.items()}
    label = {int(n): f"{float(f) * 100:.0f}%" for n, f in zip(np.asarray(counts, int), np.asarray(fracs, float))}
    rows = []
    for step in range(len(M)):
        m = M[step]; ratio = m / base if base else np.nan
        row = {"情景": scen_cn, "变化站点数": int(step), "M": round(float(m), 3),
               "M/M_base": round(float(ratio), 5), "相对改善%": round(float((1.0 - ratio) * 100.0), 3)}
        for k in ("M_access", "M_crowd", "M_reach"):
            if k in comp:
                row[k] = round(float(comp[k][step]), 3)
                cb = cbase[k]
                row[k + "/base"] = round(float(comp[k][step] / cb), 5) if (cb and np.isfinite(cb)) else np.nan
        row["比例节点"] = label.get(int(step), "")
        rows.append(row)
    return rows


def frac_summary_rows(scen, r):
    """各比例节点（论文报告口径）一行：比例、目标数、实际执行数、该点 M 与相对改善。"""
    M = np.asarray(r["M"], float)
    base = M[0] if M.size and M[0] else np.nan
    fracs = np.asarray(r.get("fracs", []), float)
    counts = np.asarray(r.get("counts", []), int)
    actual = np.asarray(r.get("actual_counts", counts), int)
    comp = {k: np.asarray(r[k], float) for k in ("M_access", "M_crowd", "M_reach")
            if k in r and len(r[k]) == len(M)}
    out = []
    for f, n_tgt, n_act in zip(fracs, counts, actual):
        n = int(min(n_act, len(M) - 1))
        ratio = M[n] / base if (base and n < len(M)) else np.nan
        row = {"情景": SCEN_CN[scen], "比例": f"{f * 100:.0f}%", "目标变化数": int(n_tgt),
               "实际执行数": int(n_act), "M": round(float(M[n]), 3),
               "M/M_base": round(float(ratio), 5), "相对改善%": round(float((1 - ratio) * 100), 3)}
        for k in ("M_access", "M_crowd", "M_reach"):
            if k in comp:
                row[k] = round(float(comp[k][n]), 3)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# 单个年份：跑 baseline + 选定策略 + 对照，落盘到 Outputs/<city>/yearly/<Y>/
# ---------------------------------------------------------------------------
def run_year(Y, ctx):
    D = ctx["D"]; w = D["w"]; lc = D["lon_c"]; ac = D["lat_c"]
    clon = ctx["clon"]; clat = ctx["clat"]; pool_n = ctx["pool_n"]
    strategies = ctx["strategies"]; backend = ctx["backend"]
    out = ctx["out_root"] / cso.CITY / "yearly" / str(Y)
    out.mkdir(parents=True, exist_ok=True)

    st = cso.station_snapshot(Y)
    if len(st) < 2:
        print(f"[{Y}] 存量 {len(st)} 站，过少，跳过")
        return None
    slon, slat = st["lon"].to_numpy(), st["lat"].to_numpy()
    fast, slow = st["fast"].to_numpy(), st["slow"].to_numpy()
    sid = st["sid"].to_numpy()
    c = cso.dist_matrix(lc, ac, slon, slat)
    s_fix = cso._disp_scale(w, fast, slow)
    b = cso.baseline_report(w, c, fast, slow, s=s_fix)
    d_ref = b["d_ref"]
    print(f"[{Y}] 存量 {len(st)} | M={b['M']:.0f}=acc{b['M_access']:.0f}+crowd{b['M_crowd']:.0f}+reach{b['M_reach']:.0f}"
          f" | 可达覆盖 {b['reach_cov']:.3f} | 超容量站 {b['over_cap']}/{len(st)}")

    # --- baseline.csv（含原生 KPI）---
    pd.DataFrame([
        {"指标": "M_2SFCA", "值": round(b["M"], 1), "单位": "需求·km"},
        {"指标": "M_access(出行)", "值": round(b["M_access"], 1), "单位": "需求·km"},
        {"指标": "M_crowd(拥挤)", "值": round(b["M_crowd"], 1), "单位": "需求·km"},
        {"指标": "M_reach(够不着)", "值": round(b["M_reach"], 1), "单位": "需求·km"},
        {"指标": "M_old(零拥挤参照)", "值": round(b["M_old"], 1), "单位": "需求·km"},
        {"指标": "可达覆盖率(原生)", "值": round(b["reach_cov"], 4), "单位": "占比"},
        {"指标": "不可达需求占比(原生)", "值": round(b["unreach_frac"], 4), "单位": "占比"},
        {"指标": "不可达需求点数", "值": int(b["n_unreach"]), "单位": "个"},
        {"指标": "不可达需求量", "值": round(b["w_unreach"], 2), "单位": "次/天"},
        {"指标": "拥挤度C中位", "值": round(b["C_med"], 3), "单位": "-"},
        {"指标": "拥挤度C90分位", "值": round(b["C_p90"], 3), "单位": "-"},
        {"指标": "超容量站(C>1)", "值": int(b["over_cap"]), "单位": f"/{len(st)}"},
        {"指标": "站点数", "值": int(len(st)), "单位": "个"},
    ]).to_csv(out / "baseline.csv", index=False, encoding="utf-8-sig")

    # --- 现状失配分解柱状图 ---
    fig, ax = plt.subplots(figsize=style.mm(100, 64))
    lab = ["可服务(access)", "拥挤(容量失配)", "够不着(空间失配)"]
    val = [b["M_access"], b["M_crowd"], b["M_reach"]]
    ax.bar(lab, val, color=[style.C["blue"], style.C["orange"], style.C["red"]], alpha=0.9)
    for i, vv in enumerate(val):
        ax.text(i, vv, f"{vv / max(sum(val), 1e-9) * 100:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("需求·km"); ax.set_title(f"{cso.CITY_NAME} · {Y} · 2SFCA 失配分解")
    _save(fig, out / "fig_baseline_decomp.png")

    # --- 策略 ---
    results = {}
    if "s1" in strategies:
        results["s1"] = cso.run_s1_add(w, c, fast, slow, clon, clat, lc, ac,
                                       s=s_fix, d_ref=d_ref, pool_n=pool_n, backend=backend)
    if "s2" in strategies:
        results["s2"] = cso.run_s2_remove(w, c, fast, slow, s=s_fix, d_ref=d_ref)
    if "s3" in strategies:
        results["s3"] = cso.run_s3_swap(w, c, fast, slow, clon, clat, lc, ac,
                                        s=s_fix, d_ref=d_ref, pool_n=pool_n, backend=backend)

    # 每年的情景曲线：1×k 子图（同一年的 S1/S2/S3，**不跨年**）
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=style.mm(82 * len(results), 74), squeeze=False)
        for axi, (scen, r) in zip(axes.ravel(), results.items()):
            cso.plot_scenario_decomp(axi, r, title=f"{Y} · {SCEN_CN[scen]}",
                                     counts=r.get("actual_counts", r.get("counts")),
                                     fracs=r.get("fracs"), sign=SCEN_SIGN.get(scen, ""))
        _save(fig, out / "fig_scenarios.png")
        # 各策略逐步曲线（含三项分解列）+ 比例节点表
        for scen, r in results.items():
            pd.DataFrame(curve_rows(SCEN_CN[scen], r, r.get("actual_counts", r.get("counts")),
                                    r.get("fracs"))).to_csv(out / f"{scen}_curve.csv",
                                                            index=False, encoding="utf-8-sig")
            pd.DataFrame(frac_summary_rows(scen, r)).to_csv(out / f"{scen}_fracs.csv",
                                                            index=False, encoding="utf-8-sig")

    # --- S1 推荐新增坐标清单 ---
    a1_lon = a1_lat = np.array([])
    if "s1" in results:
        sel = np.asarray(results["s1"]["sel"], int)
        sel = sel[sel < len(clon)]
        a1_lon, a1_lat = clon[sel], clat[sel]
        near = np.array([float(np.min(cso.haversine_km(x, y, slon, slat))) for x, y in zip(a1_lon, a1_lat)]) \
            if a1_lon.size else np.array([])
        pd.DataFrame({"序号": np.arange(1, len(a1_lon) + 1), "经度": a1_lon, "纬度": a1_lat,
                      "到最近现有站km": np.round(near, 3) if near.size else [],
                      "热点同址<1km": (near < 1.0) if near.size else []}).to_csv(
            out / "s1_add.csv", index=False, encoding="utf-8-sig")

    # --- S2 关闭清单 ---
    rm_idx = np.array([], int)
    if "s2" in results:
        rm_idx = np.asarray(results["s2"]["order"], int)
        pd.DataFrame({"删除序号": np.arange(1, len(rm_idx) + 1),
                      "station_id": sid[rm_idx] if rm_idx.size else [],
                      "经度": slon[rm_idx] if rm_idx.size else [],
                      "纬度": slat[rm_idx] if rm_idx.size else []}).to_csv(
            out / "s2_remove.csv", index=False, encoding="utf-8-sig")

    # --- S3 调配清单 ---
    if "s3" in results:
        rmv = np.asarray(results["s3"]["removed"], int)
        aidx = np.asarray(results["s3"]["added"], int); aidx = aidx[aidx < len(clon)]
        ch = [{"类型": "关闭", "station_id": str(sid[i]), "经度": float(slon[i]), "纬度": float(slat[i])} for i in rmv]
        ch += [{"类型": "新增(路网节点)", "station_id": "NEW", "经度": float(clon[j]), "纬度": float(clat[j])} for j in aidx]
        pd.DataFrame(ch).to_csv(out / "s3_swap.csv", index=False, encoding="utf-8-sig")

    # --- 该年变化地图（需求路段着色 + 现有站 + 推荐新增 + 关闭 + 真实下一年新增）---
    real = cso.real_additions_between(Y, Y + 1) if (Y + 1) <= ctx["max_year"] else None
    rl = real["lon"].to_numpy() if real is not None and len(real) else None
    ra = real["lat"].to_numpy() if real is not None and len(real) else None
    fig, ax = plt.subplots(figsize=style.mm(150, 150))
    lc_map = cso.plot_change_map(ax, D, slon, slat,
                                 add_lon=a1_lon[:80] if a1_lon.size else None,
                                 add_lat=a1_lat[:80] if a1_lat.size else None,
                                 remove_idx=rm_idx[:80] if rm_idx.size else None,
                                 real_lon=rl, real_lat=ra, norm=ctx["norm"],
                                 title=f"{cso.CITY_NAME} · {Y} 变化地图")
    if lc_map is not None:
        fig.colorbar(lc_map, ax=ax, fraction=0.04, pad=0.02).set_label("期望低电量需求 (次/天/节点)", fontsize=7)
    _save(fig, out / "fig_change_map.png")

    # --- 推荐 vs 真实下一年新增 对照（仅当有 S1 且有 Y+1）---
    hit = {}
    if "s1" in results and real is not None and len(real) > 0 and a1_lon.size:
        K = len(real)
        sel = np.asarray(results["s1"]["sel"], int); sel = sel[sel < len(clon)][:K]
        if sel.size:
            cmp = cso.compare_recommendations_to_real(clon[sel], clat[sel],
                                                      real["lon"].to_numpy(), real["lat"].to_numpy())
            hit = cmp["hit_rate"]
            pd.DataFrame({"真实新增经度": real["lon"].to_numpy(), "真实新增纬度": real["lat"].to_numpy(),
                          "到最近推荐点km": np.round(cmp["nearest_km"], 3)}).to_csv(
                out / "real_vs_rec.csv", index=False, encoding="utf-8-sig")
            d = np.sort(cmp["nearest_km"])
            fig, ax = plt.subplots(figsize=style.mm(110, 70))
            ax.plot(d, np.arange(1, len(d) + 1) / len(d), "-", color=style.C["blue"])
            for t in cso.MATCH_THRESH_KM:
                ax.axvline(t, color=style.C["gray"], ls=":", lw=0.8)
            ax.set_xlabel("真实新增到最近推荐点距离 km"); ax.set_ylabel("累计占比")
            ax.set_title(f"{cso.CITY_NAME} · {Y}→{Y+1} 推荐-真实最近距离 CDF")
            ax.set_xlim(0, max(10, float(np.percentile(d, 95))))
            _save(fig, out / "fig_real_vs_rec.png")

    # --- 汇总行（跨年只进表，不进混合图）---
    def _best(scen):
        if scen not in results:
            return np.nan
        M = np.asarray(results[scen]["M"], float)
        return round(float(M[-1] / M[0]), 4) if M.size and M[0] else np.nan
    row = {"年份": Y, "站点数": int(len(st)), "M": round(b["M"], 1),
           "M_access": round(b["M_access"], 1), "M_crowd": round(b["M_crowd"], 1),
           "M_reach": round(b["M_reach"], 1), "可达覆盖率": round(b["reach_cov"], 4),
           "不可达占比": round(b["unreach_frac"], 4), "超容量站": int(b["over_cap"]),
           "S1末比": _best("s1"), "S2末比": _best("s2"), "S3末比": _best("s3"),
           "真实新增数(Y→Y+1)": (len(real) if real is not None else 0),
           "命中率_2km": round(hit.get(2.0, np.nan), 3) if hit else np.nan}
    print(f"      已落盘 → {out}")
    return row


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    cso.configure_city(args.city, args.city_config or None)
    smoke = SMOKE or args.smoke
    out_root = Path(args.output_root)
    if smoke:
        out_root = out_root / "_smoke"          # 冒烟不覆盖正式 Outputs
    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in {"s1", "s2", "s3"}]
    if bad:
        raise ValueError(f"未知策略 {bad}；只能是 s1/s2/s3 的子集")

    years_all = cso.station_years()
    if args.years:
        want = {int(y) for y in args.years.split(",")}
        years = [y for y in years_all if y in want]
    else:
        years = years_all
    if not years:
        raise ValueError(f"没有可用年份；station_years()={years_all}")
    max_year = max(years_all)                     # 真实新增对照以全量年份判断是否存在 Y+1

    draws = (4 if smoke else cso.N_ENSEMBLE)
    print(f"\n===== {cso.CITY_NAME}({cso.CITY}) | 年份 {years} | 策略 {strategies} | "
          f"抽样 {draws} | 后端 {args.backend}{' | SMOKE' if smoke else ''} =====")
    D = cso.build_demand_surface(draws=draws, days=(cso.DAYS[:1] if smoke else None))
    w = D["w"]
    print(f"固定需求面：{D['n_cells']} 个路网节点 | 期望低电量事件合计 {w.sum():.0f} 次/天 | 天数 {D['n_days']}")

    # 候选规模按“全年份最大存量 × 最大 ADD/SWAP 比例”动态抬高（CAND_CAP/CAP_POOL 仅作加速下限）
    max_nF = max(len(cso.station_snapshot(Y)) for Y in years)
    max_strategy_n = 0
    if "s1" in strategies:
        max_strategy_n = max(max_strategy_n, max(cso.counts_from_fracs(max_nF, cso.ADD_FRACS)))
    if "s3" in strategies:
        max_strategy_n = max(max_strategy_n, max(cso.counts_from_fracs(max_nF, cso.SWAP_FRACS)))
    cand_cap = max((80 if smoke else cso.CAND_CAP), max_strategy_n)
    pool_n = max((20 if smoke else cso.CAP_POOL), max_strategy_n)
    clon = clat = np.array([])
    if {"s1", "s3"} & set(strategies):
        clon, clat = cso._candidates_from_demand(D, cap=cand_cap, smoke=smoke)
        print(f"候选新增点 {len(clon)} 个 | CELF 候选池上限 {pool_n} | 最大策略目标 {max_strategy_n}")

    # 城市级固定需求面图（需求每年不变，单独出一张，避免每年重复）
    city_dir = out_root / cso.CITY
    city_dir.mkdir(parents=True, exist_ok=True)
    norm = _demand_norm(w)
    fig, ax = plt.subplots(figsize=style.mm(150, 150))
    lc0 = cso.plot_demand_network(ax, D, norm=norm, title=f"{cso.CITY_NAME} 固定低电量需求面（路段着色）")
    if lc0 is not None:
        fig.colorbar(lc0, ax=ax, fraction=0.04, pad=0.02).set_label("期望低电量需求 (次/天/节点)", fontsize=7)
    _save(fig, city_dir / "fig_demand_network.png")

    ctx = dict(D=D, clon=clon, clat=clat, pool_n=pool_n, strategies=strategies,
               backend=args.backend, out_root=out_root, max_year=max_year, norm=norm)

    summary = []
    for Y in years:
        row = run_year(Y, ctx)
        if row is not None:
            summary.append(row)

    # 跨年汇总：只出一张表 + 一张“每年一根柱”的分项柱状图（非叠线，不混合）
    sdf = pd.DataFrame(summary)
    sdf.to_csv(city_dir / "yearly_summary.csv", index=False, encoding="utf-8-sig")
    if len(sdf):
        yrs = sdf["年份"].to_numpy()
        fig, ax = plt.subplots(figsize=style.mm(160, 86))
        x = np.arange(len(yrs))
        ax.bar(x, sdf["M_access"], color=style.C["blue"], label="M_access")
        ax.bar(x, sdf["M_crowd"], bottom=sdf["M_access"], color=style.C["orange"], label="M_crowd")
        ax.bar(x, sdf["M_reach"], bottom=sdf["M_access"] + sdf["M_crowd"], color=style.C["red"], label="M_reach")
        ax.set_xticks(x); ax.set_xticklabels([str(y) for y in yrs])
        ax.set_ylabel("需求·km"); ax.set_xlabel("年份")
        ax.set_title(f"{cso.CITY_NAME} 逐年现状错配分解（每年一根柱）"); ax.legend(loc="best")
        ax2 = ax.twinx()
        ax2.plot(x, sdf["可达覆盖率"], "-o", color=style.C["green"], lw=1.2, ms=3)
        ax2.set_ylabel("可达覆盖率(原生)", color=style.C["green"]); ax2.set_ylim(0, 1.02)
        _save(fig, city_dir / "fig_yearly_summary.png")
    print(f"\n全部完成。逐年成果在 {out_root / cso.CITY / 'yearly'}/<年份>/，汇总在 {city_dir / 'yearly_summary.csv'}")


def parse_args():
    p = argparse.ArgumentParser(description="年度版充电站 2SFCA 错配 S1/S2/S3 批处理（每年单独出成果）")
    p.add_argument("--city", default=os.environ.get("CSO_CITY", "guangzhou"))
    p.add_argument("--city-config", default=os.environ.get("CSO_CITY_CONFIG", ""))
    p.add_argument("--years", default="", help="逗号分隔年份子集，如 2019,2021；缺省=全部发现年份")
    p.add_argument("--strategies", default="s1,s2,s3", help="逗号分隔 s1/s2/s3 子集")
    p.add_argument("--backend", default="fast", choices=["dense", "fast"], help="CELF 后端，默认 fast 提高速度")
    p.add_argument("--smoke", action="store_true", help="冒烟：少抽样/少候选，写 Outputs/_smoke")
    p.add_argument("--output-root", default=str(cso.OUTPUT_ROOT))
    return p.parse_args()


if __name__ == "__main__":
    main()