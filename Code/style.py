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


def savefig(fig, path):
    """统一存盘：600dpi PNG + 同名矢量 PDF（矢量图任意放大不糊）。"""
    path = Path(path)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
