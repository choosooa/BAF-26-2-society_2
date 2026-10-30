"""공통 경로, 로더, 컬럼 그룹, 그림 헬퍼."""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EDA = ROOT / "eda"
CACHE = EDA / "cache"
RESULTS = CACHE / "results"
FIG = EDA / "figures"
for d in (CACHE, RESULTS, FIG):
    d.mkdir(parents=True, exist_ok=True)

ID = ["row_id", "timestamp", "station_id"]
WL = ["wl_t", "wl_t_minus_1h", "wl_t_minus_3h", "wl_t_minus_6h", "wl_t_minus_12h", "wl_t_minus_24h"]
WL_LAGS = {1: "wl_t_minus_1h", 3: "wl_t_minus_3h", 6: "wl_t_minus_6h", 12: "wl_t_minus_12h", 24: "wl_t_minus_24h"}
IMERG = ["imerg_rain_1h", "imerg_rain_6h", "imerg_rain_24h"]
SFC = ["sfc_rain_1h", "sfc_rain_6h", "sfc_rain_24h"]
RADAR = ["radar_reflectivity_mean", "radar_reflectivity_p95", "radar_reflectivity_ge20_fraction"]
ENV = ["cat_area", "wsarea", "mean_elevation", "mean_slope", "urban_ratio", "forest_ratio", "stream_order"]
TARGET = "target_maxrise_6h"
FEATURES = WL + IMERG + SFC + RADAR + ENV

plt.rcParams["font.family"] = ["DejaVu Sans", "AppleGothic"]  # 숫자·기호는 DejaVu, 한글은 AppleGothic으로 폴백
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def load(name, cols=None):
    """name: train | test | train_sorted | test_sorted"""
    return pd.read_parquet(CACHE / f"{name}.parquet", columns=cols)


def savefig(fig, fname):
    path = FIG / fname
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  [fig] {path.name}")
    return path


def save_json(obj, fname):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(RESULTS / fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=conv)
    print(f"  [json] {fname}")


def load_json(fname):
    with open(RESULTS / fname, encoding="utf-8") as f:
        return json.load(f)


def rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def peak_rss(tag=""):
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mb = r / 1024 / 1024 if sys.platform == "darwin" else r / 1024
    print(f"[mem] {tag} peak RSS = {mb:,.0f} MB")
    return mb


def binned_stats(x, y, q=20):
    """x 분위 구간별 y 평균·분위수. 0이 많은 x는 0을 별도 구간으로."""
    x = pd.Series(np.asarray(x, dtype=np.float64))
    y = pd.Series(np.asarray(y, dtype=np.float64))
    bins = pd.Series(pd.NA, index=x.index, dtype="object")
    zero = x == 0
    nz = ~zero
    if nz.sum() > q:
        b = pd.qcut(x[nz], q=q, duplicates="drop")
        bins[nz] = b.astype(str)
    grp_key = np.where(zero, "0", bins.astype(str))
    df = pd.DataFrame({"x": x, "y": y, "k": grp_key})
    g = df.groupby("k", sort=False).agg(
        x_mid=("x", "median"), n=("y", "size"), mean=("y", "mean"),
        q10=("y", lambda s: s.quantile(0.1)), q90=("y", lambda s: s.quantile(0.9)), std=("y", "std"),
    )
    return g.sort_values("x_mid")


def symlog_x(ax, linthresh=0.1):
    """symlog x축 + 일반 숫자 눈금 (mathtext 마이너스 글리프 문제 회피)."""
    from matplotlib.ticker import FuncFormatter
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


# ---- 탐색형 EDA(S8~S10)용 스타일: dataviz 기준 팔레트 (검증 통과: 앞 3색 all-pairs) ----
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_MID = "#f0efec"


def style():
    plt.rcParams.update({
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "lines.linewidth": 2, "legend.frameon": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    })


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bluered", ["#184f95", "#6da7ec", DIV_MID, "#f09a8f", "#b3261e"])
