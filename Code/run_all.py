# -*- coding: utf-8 -*-
"""
run_all.py — 多城市版一键运行脚本。

用法：
    python run_all.py
    python run_all.py truncated
    python run_all.py --city guangzhou truncated comprehensive
    SMOKE=1 python run_all.py --city guangzhou
    python run_all.py --city shenzhen --city-config ../city_configs.json

输出：
    Outputs/<city>/<mode>/
        baseline.csv
        s1_curve.csv / s2_curve.csv / s3_curve.csv
        s1_add_<lens>.csv / s2_remove_<lens>.csv / s3_swap_<lens>.csv
        fig_s1_curve.png + fig_s1_map_disp.png + fig_s1_map_queue.png
        fig_s2_curve.png + fig_s2_map_disp.png + fig_s2_map_queue.png
        fig_s3_curve.png + fig_s3_map_disp.png + fig_s3_map_queue.png
"""
from __future__ import annotations

import argparse
import os
import sys
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
LENSES = ("disp", "queue")
LENS_CN = {"disp": "空间置换", "queue": "时间排队"}
SCENARIO_CN = {"s1": "S1 只增", "s2": "S2 只减", "s3": "S3 等量调配"}


def frac_labels(counts):
    labels = [f"{int(f * 100)}%" for f in cso.ADD_FRACS]
    return labels[:len(counts)]


def lens_color(lens):
    return style.C["blue"] if lens == "disp" else style.C["red"]


def pad_history(values, target_len):
    """把提前停止的曲线补平成固定长度；这只影响展示，不改变已选站点。"""
    v = list(map(float, values))
    if not v:
        return [np.nan] * target_len
    if len(v) < target_len:
        v.extend([v[-1]] * (target_len - len(v)))
    return v[:target_len]


def curve_rows(scenario, lens, values, counts, labels, demand_sum):
    base = float(values[0])
    rows = []
    for step, m in enumerate(values):
        ratio = m / base if base else np.nan
        rows.append({
            "情景": scenario,
            "口径": LENS_CN[lens],
            "变化站点数": step,
            "M": round(float(m), 6),
            "M/M_base": round(float(ratio), 6),
            "相对改善%": round(float((1.0 - ratio) * 100.0), 4),
            "M_per_event": round(float(m / max(demand_sum, 1e-12)), 6),
            "是否ADD_FRACS节点": "",
        })
    for lab, n in zip(labels, counts):
        if int(n) < len(rows):
            rows[int(n)]["是否ADD_FRACS节点"] = lab
    return rows


def plot_curve(out, mode, scenario, histories, counts, labels):
    fig, ax = plt.subplots(figsize=style.mm(150, 88))
    ymax = 1.0
    ymin = 1.0
    for lens, values in histories.items():
        values = np.asarray(values, float)
        base = values[0]
        y = values / base if base else np.full_like(values, np.nan)
        x = np.arange(len(values))
        col = lens_color(lens)
        ax.plot(x, y, "-", lw=2.0, color=col, label=LENS_CN[lens])
        marker_x = [int(n) for n in counts if int(n) < len(values)]
        marker_y = [y[int(n)] for n in marker_x]
        ax.scatter(marker_x, marker_y, s=26, color=col, zorder=5)
        for lab, xx, yy in zip(labels, marker_x, marker_y):
            ax.annotate(lab, (xx, yy), textcoords="offset points", xytext=(0, 6),
                        ha="center", va="bottom", fontsize=7, color=col)
        ymax = max(ymax, float(np.nanmax(y)))
        ymin = min(ymin, float(np.nanmin(y)))
    ax.axhline(1.0, color=style.C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("变化站点数 N（括号为现有站点比例）")
    ax.set_ylabel("相对错配指数 M / M_base")
    ax.set_title(f"{cso.CITY_NAME} · {mode} · {SCENARIO_CN[scenario]}")
    xticks = [0] + [int(n) for n in counts]
    xticklabels = ["0"] + [f"{int(n)}\n({lab})" for n, lab in zip(counts, labels)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    pad = max(0.015, (ymax - ymin) * 0.10)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(loc="best")
    style.savefig(fig, out / f"fig_{scenario}_curve.png")
    plt.close(fig)


def plot_change_map(out, mode, scenario, lens, lc, ac, w, slon, slat,
                    add_idx=None, remove_idx=None):
    add_idx = np.asarray(add_idx if add_idx is not None else [], dtype=int)
    remove_idx = np.asarray(remove_idx if remove_idx is not None else [], dtype=int)
    fig, ax = plt.subplots(figsize=style.mm(150, 150))
    style.draw_admin(ax, cache_dir=cso.DATA, adcode=cso.CITY_ADMIN_ADCODE, color="#7E8795", lw=0.7)
    m = w > 0
    vmax = float(np.percentile(w[m], 98)) if np.any(m) else 1.0
    norm = mcolors.PowerNorm(gamma=0.72, vmin=0.0, vmax=max(vmax, 1e-9))
    sc = ax.scatter(lc[m], ac[m], s=8, c=w[m], cmap=style.SEQ, norm=norm, alpha=0.82,
                    linewidths=0, zorder=1)
    ax.scatter(slon, slat, s=2.2, c=style.C["gray"], marker=".", alpha=0.35,
               zorder=2, label="现有站")
    if remove_idx.size:
        ax.scatter(slon[remove_idx], slat[remove_idx], s=28, c=style.C["red"],
                   marker="x", linewidths=1.0, alpha=0.92, zorder=4, label="关闭")
    if add_idx.size:
        ax.scatter(lc[add_idx], ac[add_idx], s=92, c=style.C["green"],
                   marker="*", edgecolor="white", linewidths=0.6, zorder=5, label="新增")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("期望低电量需求 (次/天/格)", fontsize=7)
    ax.set_title(f"{cso.CITY_NAME} · {mode} · {SCENARIO_CN[scenario]} · {LENS_CN[lens]}")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", markerscale=1.2)
    style.savefig(fig, out / f"fig_{scenario}_map_{lens}.png")
    plt.close(fig)


def score_layout(lc, ac, w, slon, slat, fast, slow, s_fix):
    c2 = cso.dist_matrix(lc, ac, slon, slat)
    return {
        "M_old_km": round(cso.M_old(w, c2), 1),
        "空间置换_min": round(cso.M_disp(w, c2, fast, slow, s=s_fix)["M"] * 60.0 / cso.AVG_SPEED, 1),
        "时间排队_min": round(cso.M_queue(w, c2, fast, slow)["M"], 1),
    }


def run_mode(mode, output_root):
    out = Path(output_root) / cso.CITY / mode
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n========== 城市 = {cso.CITY_NAME}({cso.CITY}) | 口径 = {mode} ==========")
    st, stats = cso.load_stations(mode, return_stats=True)
    slon, slat = st["lon"].to_numpy(), st["lat"].to_numpy()
    fast, slow = st["fast"].to_numpy(), st["slow"].to_numpy()
    sid = st["sid"].to_numpy()
    nF = len(slon)
    print(f"站点：原始 {stats['raw']} → 保留 {nF}（晚于轨迹日 {stats['future_create']}，零容量 {stats['zero_cap']}）")

    draws = 3 if SMOKE else cso.N_ENSEMBLE
    D = cso.demand_surface(draws=draws)
    lc, ac, w = D["lon_c"], D["lat_c"], D["w"]
    print(f"需求格 {D['n_cells']} 个 | 期望低电量需求合计 {w.sum():.0f}")
    c = cso.dist_matrix(lc, ac, slon, slat)

    b = cso.both_indices(w, c, fast, slow)
    pd.DataFrame([
        {"指标": "M_old(无限容量·零拥挤参照)", "值": round(b["M_old_km"], 1), "单位": "需求·km"},
        {"指标": "空间置换 M_disp", "值": round(b["M_disp_km"], 1), "单位": "需求·km"},
        {"指标": "空间置换(折分钟)", "值": round(b["M_disp_min"], 1), "单位": "需求·min"},
        {"指标": "时间排队 M_queue", "值": round(b["M_queue_min"], 1), "单位": "需求·min"},
        {"指标": "空间置换·可达内无站抛锚", "值": round(b["disp"]["M_dead_range"], 1), "单位": "需求·km"},
        {"指标": "空间置换·容量满抛锚", "值": round(b["disp"]["M_dead_cap"], 1), "单位": "需求·km"},
        {"指标": "空间置换·每次低电量事件", "值": round(b["M_disp_km"] / max(w.sum(), 1e-12), 4), "单位": "km/次"},
        {"指标": "时间排队·每次低电量事件", "值": round(b["M_queue_min"] / max(w.sum(), 1e-12), 4), "单位": "min/次"},
    ]).to_csv(out / "baseline.csv", index=False, encoding="utf-8-sig")
    print(f"基线：M_old={b['M_old_km']:.0f}km | 空间置换={b['M_disp_km']:.0f}km | 时间排队={b['M_queue_min']:.0f}min")

    counts = cso.add_counts(nF)
    labels = frac_labels(counts)
    if SMOKE:
        counts = (max(1, min(counts[0], 3)),)
        labels = labels[:1]
    n_max = max(counts)
    cap_cap = 80 if SMOKE else cso.CAND_CAP
    cand = np.argsort(w)[::-1][:min(int(cap_cap), int((w > 0).sum()))]
    pool = min(len(cand), max(20 if SMOKE else cso.CAP_POOL, int(n_max)))
    if pool < n_max:
        print(f"提示：候选需求格只有 {pool} 个，小于目标变化站点数 {n_max}，曲线尾部会按最后可选布局补平。")

    target_len = int(n_max) + 1
    s_fix = cso._disp_scale(w, fast, slow)
    results = {"s1": {}, "s2": {}, "s3": {}}
    curve_all = {"s1": [], "s2": [], "s3": []}
    cross = [{
        "布局": "基线", "优化口径": "-", "城市": cso.CITY, "口径": mode,
        "M_old_km": round(b["M_old_km"], 1),
        "空间置换_min": round(b["M_disp_min"], 1),
        "时间排队_min": round(b["M_queue_min"], 1),
    }]

    for lens in LENSES:
        print(f"  · {LENS_CN[lens]}：S1/S2/S3")
        r1 = cso.greedy_add(w, c, fast, slow, lc[cand], ac[cand], lc, ac, n_max,
                            lens=lens, pool_n=pool, s=s_fix, force_n=True,
                            desc=f"S1 只增[{LENS_CN[lens]}]")
        r2 = cso.greedy_remove(w, c, fast, slow, n_max, lens=lens, s=s_fix)
        r3 = cso.swap(w, c, fast, slow, lc[cand], ac[cand], lc, ac, n_max,
                      lens=lens, pool_n=pool, s=s_fix)

        results["s1"][lens] = r1
        results["s2"][lens] = r2
        results["s3"][lens] = r3
        h1 = pad_history(r1["M"], target_len)
        h2 = pad_history(r2["M"], target_len)
        h3 = pad_history(r3["M"], target_len)
        curve_all["s1"].extend(curve_rows("S1只增", lens, h1, counts, labels, w.sum()))
        curve_all["s2"].extend(curve_rows("S2只减", lens, h2, counts, labels, w.sum()))
        curve_all["s3"].extend(curve_rows("S3等量调配", lens, h3, counts, labels, w.sum()))

        # 清单输出：坐标数量按 n_max，若候选不足则自然少于 n_max，并在 CSV 中可见。
        sel = np.asarray(r1["sel"], int)
        add_grid = cand[sel] if sel.size else np.array([], int)
        near = np.array([float(np.min(cso.haversine_km(x, y, slon, slat))) for x, y in zip(lc[add_grid], ac[add_grid])]) if add_grid.size else np.array([])
        pd.DataFrame({
            "序号": np.arange(1, len(add_grid) + 1),
            "经度": lc[add_grid] if add_grid.size else [],
            "纬度": ac[add_grid] if add_grid.size else [],
            "到最近现有站km": np.round(near, 3) if near.size else [],
            "热点同址<1km": (near < 1.0) if near.size else [],
        }).to_csv(out / f"s1_add_{lens}.csv", index=False, encoding="utf-8-sig")

        rm = np.asarray(r2["order"], int)
        pd.DataFrame({
            "删除序号": np.arange(1, len(rm) + 1),
            "station_id": sid[rm] if rm.size else [],
            "经度": slon[rm] if rm.size else [],
            "纬度": slat[rm] if rm.size else [],
        }).to_csv(out / f"s2_remove_{lens}.csv", index=False, encoding="utf-8-sig")

        rmv = np.asarray(r3["removed"], int)
        ad = cand[np.asarray(r3["added"], int)] if r3["added"] else np.array([], int)
        ch = [{"类型": "关闭", "station_id": str(sid[i]), "经度": float(slon[i]), "纬度": float(slat[i])} for i in rmv]
        ch += [{"类型": "新增", "station_id": "NEW", "经度": float(lc[g]), "纬度": float(ac[g])} for g in ad]
        pd.DataFrame(ch).to_csv(out / f"s3_swap_{lens}.csv", index=False, encoding="utf-8-sig")

        if len(rmv) != len(ad):
            print(f"警告：S3 {LENS_CN[lens]} 关闭 {len(rmv)}、新增 {len(ad)}，候选不足时才允许不等量。")

        # 交叉打分：只对最大规模布局做精确打分，避免运行时间膨胀。
        nf, ns = r1["new_fast"], r1["new_slow"]
        if add_grid.size:
            sc = score_layout(lc, ac, w,
                              np.concatenate([slon, lc[add_grid]]),
                              np.concatenate([slat, ac[add_grid]]),
                              np.concatenate([fast, np.full(len(add_grid), nf)]),
                              np.concatenate([slow, np.full(len(add_grid), ns)]),
                              s_fix)
            cross.append({"布局": "S1只增", "优化口径": LENS_CN[lens], "城市": cso.CITY, "口径": mode, **sc})
        if len(rmv) and len(ad):
            keep = np.ones(nF, bool)
            keep[rmv] = False
            sc = score_layout(lc, ac, w,
                              np.concatenate([slon[keep], lc[ad]]),
                              np.concatenate([slat[keep], ac[ad]]),
                              np.concatenate([fast[keep], np.full(len(ad), r3["new_fast"])]),
                              np.concatenate([slow[keep], np.full(len(ad), r3["new_slow"])]),
                              s_fix)
            cross.append({"布局": "S3等量调配", "优化口径": LENS_CN[lens], "城市": cso.CITY, "口径": mode, **sc})

    for scenario in ("s1", "s2", "s3"):
        histories = {}
        for lens in LENSES:
            histories[lens] = pad_history(results[scenario][lens]["M"], target_len)
        plot_curve(out, mode, scenario, histories, counts, labels)
        pd.DataFrame(curve_all[scenario]).to_csv(out / f"{scenario}_curve.csv", index=False, encoding="utf-8-sig")

    for lens in LENSES:
        s1_add = cand[np.asarray(results["s1"][lens]["sel"], int)] if results["s1"][lens]["sel"] else np.array([], int)
        s2_rm = np.asarray(results["s2"][lens]["order"], int)
        s3_rm = np.asarray(results["s3"][lens]["removed"], int)
        s3_add = cand[np.asarray(results["s3"][lens]["added"], int)] if results["s3"][lens]["added"] else np.array([], int)
        plot_change_map(out, mode, "s1", lens, lc, ac, w, slon, slat, add_idx=s1_add)
        plot_change_map(out, mode, "s2", lens, lc, ac, w, slon, slat, remove_idx=s2_rm)
        plot_change_map(out, mode, "s3", lens, lc, ac, w, slon, slat, add_idx=s3_add, remove_idx=s3_rm)

    pd.DataFrame(cross).to_csv(out / "cross_score.csv", index=False, encoding="utf-8-sig")
    print(f"已落盘 → {out}")


def parse_args():
    p = argparse.ArgumentParser(description="多城市充电站错配指数 S1/S2/S3 一键运行")
    p.add_argument("modes", nargs="*", default=["truncated", "comprehensive"],
                   help="运行口径：truncated comprehensive；不填则两个都跑")
    p.add_argument("--city", default=os.environ.get("CSO_CITY", "guangzhou"),
                   help="城市配置键，例如 guangzhou、shenzhen。默认 guangzhou。")
    p.add_argument("--city-config", default=os.environ.get("CSO_CITY_CONFIG", ""),
                   help="外部城市配置 JSON，支持全国多城市数据文件、bbox、adcode。")
    p.add_argument("--output-root", default=str(cso.ROOT / "Outputs"),
                   help="输出根目录；最终路径为 <output-root>/<city>/<mode>/。")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cso.configure_city(args.city, args.city_config or None)
    valid_modes = {"truncated", "comprehensive"}
    modes = args.modes or ["truncated", "comprehensive"]
    bad = [m for m in modes if m not in valid_modes]
    if bad:
        raise ValueError(f"未知口径：{bad}；只能是 truncated / comprehensive")
    for mode in modes:
        run_mode(mode, args.output_root)
    print("\n全部完成。")
