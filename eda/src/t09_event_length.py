"""호우사상 길이 계산 + 분포 그림.

사상 = 전체 합성 시각 축에서 1시간 넘는 공백으로 끊긴 구간 (AGENT_REFERENCE 7절).
길이 두 가지:
  (1) 사상 길이  = 사상 마지막 시각 − 첫 시각 + 1h (전체 시각 축 기준)
  (2) 관측소-사상 길이 = 한 관측소가 그 사상에서 가진 행 수 (빠진 시각 제외)
분석용만. 사상 길이는 미래 정보라 모델 변수로 쓰지 않는다.
실행: python eda/src/t09_event_length.py (프로젝트 루트에서)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eda" / "figures" / "event_length"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def add_event_id(df):
    data = df[["station_id", "timestamp"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    ts = pd.Series(sorted(data["timestamp"].unique()))
    event_map = dict(zip(ts, (ts.diff() > pd.Timedelta("1h")).cumsum()))
    data["event_id"] = data["timestamp"].map(event_map)
    return data


def lengths(data):
    ev = data.groupby("event_id")["timestamp"].agg(["min", "max"])
    ev["len_h"] = ((ev["max"] - ev["min"]) / pd.Timedelta("1h")).astype(int) + 1
    ev["n_station"] = data.groupby("event_id")["station_id"].nunique()
    ev["n_rows"] = data.groupby("event_id").size()
    se = data.groupby(["event_id", "station_id"])["timestamp"].agg(["min", "max", "size"])
    se["span_h"] = ((se["max"] - se["min"]) / pd.Timedelta("1h")).astype(int) + 1
    se = se.rename(columns={"size": "n_rows"})
    return ev, se


def desc(s):
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    return {"n": int(s.size), "mean": round(float(s.mean()), 1), "min": int(s.min()),
            "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4],
            "max": int(s.max()), "skew": round(float(s.skew()), 2)}


res = {}
evs, ses = {}, {}
for name in ["train", "test"]:
    df = pd.read_csv(ROOT / "data" / f"{name}.csv", usecols=["station_id", "timestamp"])
    data = add_event_id(df)
    ev, se = lengths(data)
    evs[name], ses[name] = ev, se
    ev.to_csv(OUT / f"event_length_{name}.csv")
    # 행 가중: 한 행이 속한 사상 길이 (RMSE는 행 단위라 행 기준 분포도 본다)
    row_len = data["event_id"].map(ev["len_h"])
    res[name] = {
        "사상_길이_h": desc(ev["len_h"]),
        "관측소사상_행수": desc(se["n_rows"]),
        "관측소사상_구간_h": desc(se["span_h"]),
        "행기준_사상길이_h": desc(row_len),
        "사상_관측소수": desc(ev["n_station"]),
        "rho_길이_관측소수": round(float(ev["len_h"].corr(ev["n_station"], method="spearman")), 3),
        "상위10%사상_행비율": round(float(ev.nlargest(max(1, len(ev) // 10), "len_h")["n_rows"].sum() / ev["n_rows"].sum()), 3),
        "빠진시각비율": round(float(1 - se["n_rows"].sum() / se["span_h"].sum()), 4),
    }
(OUT / "event_length_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
print(json.dumps(res, ensure_ascii=False, indent=2))

# ---- 그림 1: 사상 길이 분포 (train/test) ----
c = {"train": "#2a6fb0", "test": "#e07b39"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
bins = np.arange(0, 160, 4)
for n in ["train", "test"]:
    ax[0].hist(evs[n]["len_h"], bins=bins, alpha=0.6, color=c[n],
               label=f"{n} ({len(evs[n])}개, 중앙값 {evs[n]['len_h'].median():.0f}h)")
ax[0].set(title="사상 길이 분포 (사상 1개 = 1)", xlabel="사상 길이 (시간)", ylabel="사상 수")
ax[0].legend()
lb = np.logspace(0, np.log10(200), 25)
for n in ["train", "test"]:
    ax[1].hist(evs[n]["len_h"], bins=lb, alpha=0.6, color=c[n], label=n)
ax[1].set_xscale("log")
ax[1].set(title="사상 길이 분포 (로그 x축)", xlabel="사상 길이 (시간, 로그)", ylabel="사상 수")
ax[1].legend()
for n in ["train", "test"]:
    s = np.sort(evs[n]["len_h"].values)
    ax[2].step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=c[n], label=n)
ax[2].axvline(6, color="gray", ls="--", lw=1)
ax[2].text(6.5, 0.05, "6h (타깃 창)", color="gray")
ax[2].set_xscale("log")
ax[2].set(title="사상 길이 누적분포", xlabel="사상 길이 (시간, 로그)", ylabel="누적 비율")
ax[2].legend()
fig.tight_layout()
fig.savefig(OUT / "el1_event_length_dist.png", dpi=130)

# ---- 그림 2: 관측소-사상 행 수 + 행 기준 분포 + 길이 vs 관측소 수 ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for n in ["train", "test"]:
    ax[0].hist(ses[n]["n_rows"], bins=lb, alpha=0.6, color=c[n],
               weights=np.full(len(ses[n]), 1 / len(ses[n])), label=f"{n} (중앙값 {ses[n]['n_rows'].median():.0f}행)")
ax[0].set_xscale("log")
ax[0].set(title="관측소-사상 길이 (행 수)", xlabel="행 수 (로그)", ylabel="관측소-사상 비율")
ax[0].legend()
for n in ["train", "test"]:
    ev = evs[n]
    ax[1].hist(ev["len_h"], bins=lb, weights=ev["n_rows"] / ev["n_rows"].sum(),
               alpha=0.6, color=c[n], label=n)
ax[1].set_xscale("log")
ax[1].set(title="행 기준: 각 행이 속한 사상의 길이", xlabel="사상 길이 (시간, 로그)", ylabel="행 비율")
ax[1].legend()
for n in ["train", "test"]:
    ax[2].scatter(evs[n]["len_h"], evs[n]["n_station"], s=18, alpha=0.7, color=c[n], label=n)
ax[2].set_xscale("log")
ax[2].set(title="사상 길이 vs 참여 관측소 수", xlabel="사상 길이 (시간, 로그)", ylabel="관측소 수")
ax[2].legend()
fig.tight_layout()
fig.savefig(OUT / "el2_station_event_and_rows.png", dpi=130)
print("그림 저장:", OUT)
