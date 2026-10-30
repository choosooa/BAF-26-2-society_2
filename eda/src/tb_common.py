"""교과서형 EDA(t*.py) 공용: 보고서 그림 스타일, 번호 규칙, 조석·사상 플래그."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (EDA, RESULTS, load, save_json, load_json, style as _base_style,
                    CAT, INK, INK2, MUTED, GRID, BASE, SEQ, div_cmap,
                    ID, WL, IMERG, SFC, RADAR, ENV, TARGET, FEATURES)

FIG_TB = EDA / "figures" / "textbook"
FIG_TB.mkdir(parents=True, exist_ok=True)

# 층화 색: 전체 = 회색, 비조석 = 파랑, 조석 = 주황
C_ALL, C_NONTIDAL, C_TIDAL = "#9a9890", CAT[0], CAT[1]

# 6.3 기준(dwl_1h 자기상관 12~13h > 0.6, 6~7h < 0)으로 확정한 강한 조석 15곳
TIDAL = ["ST0358", "ST0292", "ST0326", "ST0347", "ST0159", "ST0110", "ST0313", "ST0415",
         "ST0234", "ST0129", "ST0192", "ST0051", "ST0130", "ST0164", "ST0123"]
WEAK_TIDAL = ["ST0151", "ST0193", "ST0226", "ST0229", "ST0260"]
OPERATED_24H = ["ST0194", "ST0200"]
STUCK = ["ST0027"]


def style(size="report"):
    """size: report(A4 폭 16cm) | slide(글자 크게)."""
    _base_style()
    fs = 9 if size == "report" else 13
    plt.rcParams.update({"font.size": fs, "axes.titlesize": fs + 1, "axes.labelsize": fs,
                         "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1, "legend.fontsize": fs - 1})


def figsize(w=16, h=7):
    """cm → inch."""
    return (w / 2.54, h / 2.54)


def save(fig, num, slug):
    """그림 번호 규칙: fig{장}-{순번}_{slug}.png (300dpi)."""
    path = FIG_TB / f"fig{num}_{slug}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}")
    return path


def save_table(df, num, slug):
    path = FIG_TB / f"tab{num}_{slug}.csv"
    df.to_csv(path, encoding="utf-8-sig")
    print(f"  [tab] {path.name}")
    return path


def add_flags(df):
    """station_id 기반 층화 플래그 추가."""
    s = df["station_id"].astype(str)
    df["is_tidal"] = s.isin(TIDAL)
    df["is_stuck"] = s.isin(STUCK)
    df["st_group"] = np.select(
        [s.isin(TIDAL), s.isin(WEAK_TIDAL), s.isin(OPERATED_24H), s.isin(STUCK)],
        ["조석", "약한 조석", "24h 조작", "고정"], default="일반")
    return df


def load_tb(name="train_sorted", cols=None):
    """정렬 캐시(event_block, pos_in_block, dwl_* 포함) + 플래그."""
    if cols is not None and "station_id" not in cols:
        cols = ["station_id"] + list(cols)
    return add_flags(load(name, cols))


def station_table(df):
    """관측소 단위(n=465) 정적 변수 표."""
    return (df.groupby("station_id", observed=True)[ENV].first()
              .pipe(lambda t: add_flags(t.reset_index()).set_index("station_id")))
