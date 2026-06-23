# -*- coding: utf-8 -*-
"""
style.py — Nature 子刊风格的绘图统一设置 + 城市行政区边界叠加（不依赖 geopandas）。

用法：
    import style; style.set_nature()           # 在每个绘图脚本/笔记本开头调一次
    style.draw_admin(ax, adcode=city_adcode)     # 在地图轴上叠加指定城市二级行政区边界
    style.savefig(fig, path)                    # 统一存 300dpi PNG + 矢量 PDF
颜色： style.C['blue'] 等。
"""
import json
import urllib.request
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Nature 子刊配色（色盲友好、低饱和）
C = dict(blue="#3B6FB6", red="#C0392B", green="#2E8B6E", orange="#E08214",
         purple="#7E5AA2", gray="#5B6573", light="#D9DEE5", ink="#1A2230")
SEQ = "rocket_r" if "rocket_r" in plt.colormaps() else "OrRd"   # 顺序色阶


def set_nature(base=9.5):
    """Nature 子刊风格：无衬线、细线、去顶右框、600dpi。**中文字体放最前**避免缺字警告。"""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # 把含中文的字体放最前：matplotlib 用第一个可用字体渲染全部文字，避免“Glyph missing”警告
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "PingFang SC",
                            "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": base, "axes.titlesize": base + 1.5, "axes.labelsize": base,
        "xtick.labelsize": base - 1.5, "ytick.labelsize": base - 1.5, "legend.fontsize": base - 1.5,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=[C["blue"], C["red"], C["green"], C["orange"], C["purple"], C["gray"]]),
    })


FIGSCALE = 1.6   # 全局放大系数：Nature 标准毫米尺寸偏小，统一放大便于查看（不影响 600dpi 清晰度）

def mm(w, h):
    """毫米 → 英寸的 figsize，并按 FIGSCALE 统一放大（Nature 单栏≈89mm，双栏≈183mm）。"""
    return (w / 25.4 * FIGSCALE, h / 25.4 * FIGSCALE)


def _polys(geom):
    """从 GeoJSON geometry 抽出所有外环坐标列表（Polygon / MultiPolygon）。"""
    t = geom.get("type"); co = geom.get("coordinates", [])
    if t == "Polygon":
        return [np.asarray(co[0])]
    if t == "MultiPolygon":
        return [np.asarray(p[0]) for p in co]
    return []


def load_admin(cache_dir, adcode):
    """下载/缓存指定城市二级行政区 GeoJSON（DataV），返回 [(name, ndarray Nx2), ...]。失败返回 []。"""
    if not adcode:
        return []
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / f"admin_{adcode}.json"
    try:
        if not fp.exists():
            url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
            with urllib.request.urlopen(url, timeout=20) as r:
                fp.write_bytes(r.read())
        data = json.loads(fp.read_text(encoding="utf-8"))
        out = []
        for f in data.get("features", []):
            name = (f.get("properties") or {}).get("name", "")
            for ring in _polys(f.get("geometry") or {}):
                out.append((name, ring))
        return out
    except Exception:
        return []


def draw_admin(ax, cache_dir=None, adcode=None, color="#8A93A0", lw=0.5, label_districts=False, fontsize=5.5):
    """在地图轴上叠加指定城市的二级行政区边界（淡灰细线）。未配置 adcode 或无网络/缓存时静默跳过。"""
    if adcode is None:
        try:
            import cso
            adcode = getattr(cso, "CITY_ADMIN_ADCODE", "")
        except Exception:
            adcode = ""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "data"
    rings = load_admin(cache_dir, adcode)
    seen = set()
    for name, ring in rings:
        ax.plot(ring[:, 0], ring[:, 1], color=color, lw=lw, zorder=1, alpha=0.9)
        if label_districts and name and name not in seen:
            seen.add(name)
            ax.text(ring[:, 0].mean(), ring[:, 1].mean(), name, fontsize=fontsize,
                    color="#333", ha="center", va="center", zorder=2, alpha=0.8)
    return bool(rings)


def draw_network(ax, cache_dir=None, city=None, color="#c9d2db", lw=0.25, alpha=0.55, max_edges=None):
    """叠加 OSMnx 路网边（WGS84 经纬度，淡线）作为“需求映射到路网节点”图的底图。
    需已缓存 data/_graph_<city>.graphml（首次由 cso 联网构建）；未装 osmnx / 无缓存时静默跳过返回 False。"""
    try:
        import osmnx as ox
    except Exception:
        return False
    if city is None or cache_dir is None:
        try:
            import cso
            city = city or cso.CITY
            cache_dir = cache_dir or cso.DATA
        except Exception:
            cache_dir = cache_dir or (Path(__file__).resolve().parent.parent / "data")
    fp = Path(cache_dir) / f"_graph_{city}.graphml"
    if not fp.exists():
        return False
    try:
        G = ox.load_graphml(fp)
        n = 0
        for u, v, d in G.edges(data=True):
            geom = d.get("geometry")
            if geom is not None:
                xs, ys = geom.xy
                ax.plot(list(xs), list(ys), color=color, lw=lw, alpha=alpha, zorder=1, solid_capstyle="round")
            else:
                ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]],
                        color=color, lw=lw, alpha=alpha, zorder=1)
            n += 1
            if max_edges is not None and n > max_edges:
                break
        return True
    except Exception:
        return False


def _graph_node_key(n):
    """GraphML 读回来的节点 id 可能是 str 或 int；统一成可比对的字符串。"""
    try:
        return str(int(n))
    except Exception:
        return str(n)


def draw_network_demand(ax, node_ids, weights, cache_dir=None, city=None, cmap=None, norm=None,
                        base_color="#E2E7EE", base_lw=0.18, demand_lw=0.75,
                        base_alpha=0.34, demand_alpha=0.92, max_edges=None):
    """按路网节点需求给路段着色。
    需求面已经 snap 到路网节点，因此不再画需求散点；每条边取两端节点需求的较大值作为路段颜色。
    返回可用于 colorbar 的 LineCollection；失败或无缓存时返回 None。"""
    try:
        import osmnx as ox
    except Exception:
        return None
    if city is None or cache_dir is None:
        try:
            import cso
            city = city or cso.CITY
            cache_dir = cache_dir or cso.DATA
        except Exception:
            cache_dir = cache_dir or (Path(__file__).resolve().parent.parent / "data")
    fp = Path(cache_dir) / f"_graph_{city}.graphml"
    if not fp.exists():
        return None
    node_ids = np.asarray(node_ids)
    weights = np.asarray(weights, float)
    demand = {_graph_node_key(n): float(w) for n, w in zip(node_ids, weights)
              if np.isfinite(w) and w > 0}
    try:
        G = ox.load_graphml(fp)
        base_segments, demand_segments, demand_values = [], [], []
        for n, (u, v, d) in enumerate(G.edges(data=True)):
            geom = d.get("geometry")
            if geom is not None:
                xs, ys = geom.xy
                seg = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
            else:
                seg = np.array([[float(G.nodes[u]["x"]), float(G.nodes[u]["y"])],
                                [float(G.nodes[v]["x"]), float(G.nodes[v]["y"])]])
            val = max(demand.get(_graph_node_key(u), 0.0), demand.get(_graph_node_key(v), 0.0))
            if val > 0:
                demand_segments.append(seg); demand_values.append(val)
            else:
                base_segments.append(seg)
            if max_edges is not None and n + 1 >= max_edges:
                break
        if base_segments:
            ax.add_collection(LineCollection(base_segments, colors=base_color, linewidths=base_lw,
                                             alpha=base_alpha, zorder=1, capstyle="round"))
        if not demand_segments:
            ax.autoscale_view()
            return None
        lc = LineCollection(demand_segments, cmap=plt.get_cmap(cmap or SEQ), norm=norm,
                            linewidths=demand_lw, alpha=demand_alpha, zorder=2, capstyle="round")
        lc.set_array(np.asarray(demand_values, float))
        ax.add_collection(lc)
        ax.autoscale_view()
        return lc
    except Exception:
        return None


def savefig(fig, path):
    """统一存盘：只存 600dpi PNG（按当前需求不再输出 PDF）。"""
    path = Path(path)
    fig.savefig(path, dpi=600, bbox_inches="tight")
