"""노트북 생성기 공용: 준비 셀 코드, 셀 추가 헬퍼, 저장·실행."""
import subprocess
from pathlib import Path

import nbformat as nbf

EDA = Path(__file__).resolve().parents[1]

SETUP = r'''
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = ["DejaVu Sans", "AppleGothic"]      # 숫자는 DejaVu, 한글은 AppleGothic
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False})

EDA = Path.cwd() if (Path.cwd() / "cache").exists() else Path.cwd().parent
CACHE, RESULTS = EDA / "cache", EDA / "cache" / "results"
TARGET = "target_maxrise_6h"
WL = ["wl_t", "wl_t_minus_1h", "wl_t_minus_3h", "wl_t_minus_6h", "wl_t_minus_12h", "wl_t_minus_24h"]
DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]
IMERG = ["imerg_rain_1h", "imerg_rain_6h", "imerg_rain_24h"]
SFC = ["sfc_rain_1h", "sfc_rain_6h", "sfc_rain_24h"]
RADAR = ["radar_reflectivity_mean", "radar_reflectivity_p95", "radar_reflectivity_ge20_fraction"]
ENV = ["cat_area", "wsarea", "mean_elevation", "mean_slope", "urban_ratio", "forest_ratio", "stream_order"]
FEATURES = WL + IMERG + SFC + RADAR + ENV

# 별책 tidal_analysis.ipynb에서 판정한 특수 관측소 (dwl_1h 자기상관 기준, 타깃 미사용)
TIDAL = ["ST0358", "ST0292", "ST0326", "ST0347", "ST0159", "ST0110", "ST0313", "ST0415",
         "ST0234", "ST0129", "ST0192", "ST0051", "ST0130", "ST0164", "ST0123"]
WEAK_TIDAL = ["ST0151", "ST0193", "ST0226", "ST0229", "ST0260"]
OPERATED_24H = ["ST0194", "ST0200"]
STUCK = ["ST0027"]
SPECIAL = TIDAL + WEAK_TIDAL + OPERATED_24H + STUCK

# 색 약속: 회색 = 전체, 파랑 = 비조석(일반), 주황 = 조석/강조, 노랑 = 강우
C_ALL, C_NT, C_TIDAL, C_RAIN, C_GREEN, C_PINK, C_PURPLE = "#9a9890", "#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#e87ba4", "#4a3aa7"
CAT = [C_NT, C_TIDAL, C_GREEN, C_RAIN, C_PINK, "#008300", C_PURPLE, "#e34948"]


def keyed(d):
    """(관측소, 시각) → 정수 키 하나. 예: 관측소 12, 시각 345 → 12_000_345"""
    return d["station_code"].to_numpy(np.int64) * 1_000_000 + d["hour"].to_numpy(np.int64)


def shift_lookup(d, key, col, dh):
    """같은 관측소의 (지금 + dh)시간 col 값. 그 행이 없으면 NaN. dh < 0 이면 과거, dh > 0 이면 미래."""
    return pd.Series(d[col].to_numpy(np.float64), index=key).reindex(key + dh).to_numpy()


def station_corr(d, x, y, min_n=50, by="station_id"):
    """그룹(기본: 관측소)별 피어슨 상관을 합계 통계로 한 번에 계산."""
    t = pd.DataFrame({"s": d[by].to_numpy(), "x": x, "y": y}).dropna()
    t["xy"], t["xx"], t["yy"] = t.x * t.y, t.x * t.x, t.y * t.y
    g = t.groupby("s").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxy=("xy", "sum"), sxx=("xx", "sum"), syy=("yy", "sum"))
    cov = g.sxy / g.n - (g.sx / g.n) * (g.sy / g.n)
    return (cov / np.sqrt((g.sxx / g.n - (g.sx / g.n) ** 2) * (g.syy / g.n - (g.sy / g.n) ** 2))).where(g.n > min_n)


def symlog_x(ax, linthresh=0.1):
    """0 근처는 선형, 멀리는 로그인 x축"""
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def binned(x, y, q=20):
    """x를 구간으로 나눠 y의 평균·표준편차. x가 0인 행은 따로 한 칸."""
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    k = pd.Series("0", index=x.index, dtype=object)
    nz = x != 0
    if nz.sum() > q:
        k[nz] = pd.qcut(x[nz], q, duplicates="drop").astype(str)
    return (pd.DataFrame({"x": x, "y": y, "k": k}).groupby("k")
              .agg(x_mid=("x", "median"), n=("y", "size"), mean=("y", "mean"), std=("y", "std")).sort_values("x_mid"))


def add_flags(d):
    d["station_id"] = d["station_id"].astype(str)
    d["group"] = np.select([d.station_id.isin(TIDAL), d.station_id.isin(WEAK_TIDAL), d.station_id.isin(OPERATED_24H),
                            d.station_id.isin(STUCK)], ["조석", "약한 조석", "24h 주기", "고정"], "일반")
    d["nontidal"] = ~d.station_id.isin(TIDAL + STUCK)          # 보고서 기준 '비조석' = 조석 15곳과 고정 1곳을 뺀 나머지
    return d
'''


class Book:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))

    def setup(self, extra=""):
        self.md("## 0. 준비\n\n그림 설정, 색 약속, 특수 관측소 목록(조석 등), 도우미 함수를 불러와요. 색 약속: ⚪ 회색 = 전체 · 🔵 파랑 = 비조석(일반) · 🟠 주황 = 조석/강조 · 🟡 노랑 = 강우")
        self.code(SETUP + ("\n" + extra.strip() if extra else ""))

    def write(self, name, execute=True, timeout=2400):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        out = EDA / name
        nbf.write(nb, out)
        print("written", out, len(self.cells), "cells")
        if execute:
            r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(out),
                                f"--ExecutePreprocessor.timeout={timeout}"], capture_output=True, text=True, cwd=EDA)
            print(r.stderr[-1500:])
        return out


# ── 7~10권(선행연구 가설 확인)용 도우미. Book.setup(extra=RESEARCH_HELPERS)로 넣는다. 1~6권 SETUP은 그대로 ──
RESEARCH_HELPERS = r'''
from scipy import stats

def load_resid(d):
    """기준 모델(HGB, 사상 단위 5-fold) OOF 예측을 붙이고 남은 오차 resid = 타깃 − 예측을 만든다."""
    o = pd.read_parquet(CACHE / "oof_s09.parquet", columns=["station_code", "hour", "oof_base"])
    d = d.merge(o, on=["station_code", "hour"], how="left", validate="one_to_one")
    d["resid"] = d[TARGET] - d["oof_base"]
    return d

def phase_of(d):
    """6권 국면 정의(입력만 사용)."""
    rain = (d.sfc_rain_6h >= .5) | (d.imerg_rain_6h >= .5)
    mv = np.select([d.dwl_1h > .01, d.dwl_1h < -.01], ["상승 중", "하강 중"], "반응 없음")
    ph = np.where(rain, "비·" + pd.Series(mv, index=d.index).str.replace("반응 없음", "아직 반응 없음"), "무강우·" + pd.Series(mv, index=d.index))
    ph = np.where(d.station_id.isin(TIDAL + WEAK_TIDAL), "조석(강·약)", ph)
    ph = np.where(d.station_id.isin(OPERATED_24H + STUCK), "24h 주기·고정", ph)
    return pd.Series(ph, index=d.index)

def _edges(x, q=20):
    """0은 따로 한 칸, 나머지는 분위 경계."""
    nz = x[(x != 0) & np.isfinite(x)]
    return np.unique(np.quantile(nz, np.linspace(0, 1, q + 1)[1:-1])) if len(nz) else np.array([])

def _codes(x, edges):
    c = np.searchsorted(edges, x, side="right") + 1
    return np.where(x == 0, 0, c)

def resid_gain(x, e, ev, q=20):
    """지표 1 · 설명 가능 몫(%): 한쪽 사상 절반에서 칸 평균 잔차를 구해 다른 절반의 잔차 제곱합이 얼마나 줄어드는지. 두 방향 평균."""
    x, e, ev = np.asarray(x, float), np.asarray(e, float), np.asarray(ev)
    ok = np.isfinite(x) & np.isfinite(e); x, e, ev = x[ok], e[ok], ev[ok]
    half = ev % 2; out = []
    for a in (0, 1):
        A, B = half == a, half != a
        ed = _edges(x[A], q); ca, cb = _codes(x[A], ed), _codes(x[B], ed)
        m = pd.Series(e[A]).groupby(ca).mean()
        corr = pd.Series(cb).map(m).fillna(0).to_numpy()
        out.append(1 - ((e[B] - corr) ** 2).sum() / (e[B] ** 2).sum())
    return 100 * np.mean(out)

def event_consistency(x, e, ev, min_n=200):
    """지표 2 · 사상 일관성: 사상별 Spearman(x, e) 부호가 전체 부호와 같은 사상 비율."""
    t = pd.DataFrame({"x": np.asarray(x, float), "e": np.asarray(e, float), "ev": np.asarray(ev)}).dropna()
    s0 = np.sign(stats.spearmanr(t.x, t.e)[0])
    r = [stats.spearmanr(g.x, g.e)[0] for _, g in t.groupby("ev") if len(g) >= min_n and g.x.nunique() > 1]
    r = np.array([v for v in r if np.isfinite(v)])
    return (np.sign(r) == s0).mean() if len(r) else np.nan, len(r), s0

def group_spearman(d, x, y, by="station_id", min_n=50):
    """그룹(관측소·사상)별 Spearman: 그룹 안 순위로 바꾼 뒤 피어슨."""
    t = pd.DataFrame({"g": d[by].to_numpy(), "x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    t["rx"] = t.groupby("g").x.rank(); t["ry"] = t.groupby("g").y.rank()
    t = t.rename(columns={"g": by})
    return station_corr(t, t.rx.to_numpy(), t.ry.to_numpy(), min_n=min_n, by=by)

def share_same_sign(r, expected=1):
    """일치율: 값이 있는 그룹 중 부호가 기대 방향과 같은 비율과 그룹 수."""
    r = pd.Series(r).dropna()
    return (np.sign(r) == expected).mean() if len(r) else np.nan, len(r)

def verdict(overall_sign, expected, share):
    """0.4 판정 기준."""
    if not np.isfinite(share): return "⏸ 보류"
    if overall_sign != expected or share < .5: return "❌ 기각"
    return "✅ 채택" if share >= .7 else "🔶 부분"

def resid_check(name, x, d, mask=None):
    """후보 변수 하나의 잔차 점검 한 줄."""
    m = np.ones(len(d), bool) if mask is None else np.asarray(mask)
    g = resid_gain(np.asarray(x)[m], d.resid.to_numpy()[m], d.event_block.to_numpy()[m])
    top5 = d[m].groupby("event_block").resid.apply(lambda v: (v ** 2).sum()).nlargest(5).index
    k = m & ~d.event_block.isin(top5).to_numpy()
    g5 = resid_gain(np.asarray(x)[k], d.resid.to_numpy()[k], d.event_block.to_numpy()[k])
    c, n, s = event_consistency(np.asarray(x)[m], d.resid.to_numpy()[m], d.event_block.to_numpy()[m])
    return {"후보": name, "행": int(m.sum()), "설명 가능 몫 %": round(g, 2), "상위5사상 제외 %": round(g5, 2),
            "사상 일관성": round(c, 2) if np.isfinite(c) else np.nan, "사상 수": n, "잔차 부호": int(s)}

def lookup_past(d, key, col, k):
    """같은 관측소 k시간 전 값(없으면 NaN)."""
    return shift_lookup(d, key, col, -k)
'''
