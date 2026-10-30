# ============================================================
# 0. 설정 (스타일 / 한글 폰트 / 상수)
# ============================================================
import os
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, spearmanr

sns.set_theme(style="whitegrid")
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10

C_TR, C_TE = "#2F6DB5", "#E07B39"   # train / test 두 색만 쓴다
TIDAL_PATH = "cache/results/nb07_regimes.csv" if os.path.exists("cache/results") else "eda/cache/results/nb07_regimes.csv"   # 조석·고정 관측소 목록 (없으면 층화 생략)


# ============================================================
# 1. 사상 ID 붙이기 (AGENT_REFERENCE 7절: 전체 시각 축에서 1h 넘게 비면 새 사상)
# ============================================================
def add_event_id(frame):
    data = frame.sort_values(["station_id", "timestamp"]).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    ts = pd.Series(sorted(data["timestamp"].unique()))
    event_map = dict(zip(ts, (ts.diff() > pd.Timedelta("1h")).cumsum()))
    data["event_id"] = data["timestamp"].map(event_map)
    return data


# ============================================================
# 2. 길이 계산 (사상 길이 / 관측소-사상 길이)
# ============================================================
def event_lengths(data):
    ev = data.groupby("event_id").agg(start=("timestamp", "min"), end=("timestamp", "max"),
                                      n_rows=("row_id", "size"), n_station=("station_id", "nunique"))
    ev["length_h"] = (ev["end"] - ev["start"]) / pd.Timedelta("1h") + 1   # 끝 − 시작 + 1h
    se = data.groupby(["event_id", "station_id"]).agg(n=("timestamp", "size"),
                                                      t0=("timestamp", "min"), t1=("timestamp", "max"))
    se["span_h"] = (se["t1"] - se["t0"]) / pd.Timedelta("1h") + 1
    return ev, se


def summarize(name, s):
    q = s.quantile([.1, .25, .5, .75, .9])
    print(f"[{name}] n={len(s):,}  평균 {s.mean():.1f}  p10 {q[.1]:.0f} / p25 {q[.25]:.0f} / "
          f"중앙 {q[.5]:.0f} / p75 {q[.75]:.0f} / p90 {q[.9]:.0f}  최대 {s.max():.0f}  왜도 {skew(s):.1f}")


# ============================================================
# 3. 사상 길이 분포 (선형 / 로그 / 누적)
# ============================================================
def plot_event_length(sets):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, (ev, _), c in sets:
        L = ev["length_h"]
        axes[0].hist(L, bins=np.arange(0, L.max() + 5, 5), alpha=.55, color=c,
                     label=f"{name} ({len(L)}개)", edgecolor="white")
        axes[1].hist(L, bins=np.logspace(0, np.log10(160), 25), alpha=.55, color=c,
                     label=name, edgecolor="white")
        x = np.sort(L)
        axes[2].step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, lw=2, label=name)
        axes[0].axvline(L.median(), color=c, ls="--", lw=1.2)
    axes[1].set_xscale("log")
    axes[2].axvline(6, color="gray", ls=":", lw=1.2)
    axes[2].text(6.5, .05, "타깃 창 6h", color="gray")
    axes[2].set_xscale("log")
    titles = ["사상 길이 분포 (점선 = 중앙값)", "사상 길이 분포 (로그 축)", "사상 길이 누적분포"]
    ylabels = ["사상 수", "사상 수", "누적 비율"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t)
        ax.set_xlabel("사상 길이 (시간)")
        ax.set_ylabel(yl)
        ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. 관측소-사상 길이 / 행 기준 길이 / 관측소 수
# ============================================================
def plot_station_event(sets):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, (ev, se), c in sets:
        axes[0].hist(se["n"], bins=np.logspace(0, np.log10(160), 25), alpha=.55, color=c,
                     label=f"{name} ({len(se):,}개)", edgecolor="white")
        # 행 하나가 속한 사상의 길이 → 긴 사상이 행을 얼마나 차지하는지
        x = ev["length_h"].sort_values()
        w = ev.loc[x.index, "n_rows"].cumsum() / ev["n_rows"].sum()
        axes[1].step(x.values, w.values, where="post", color=c, lw=2, label=name)
        axes[2].scatter(ev["length_h"], ev["n_station"], s=22, alpha=.7, color=c, label=name)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[2].set_xscale("log")
    axes[0].set(title="관측소-사상 길이 (로그 축)", xlabel="관측소가 그 사상에서 가진 행 수", ylabel="관측소-사상 수")
    axes[1].set(title="행 기준 누적분포: 행이 어느 길이의 사상에 속하나", xlabel="사상 길이 (시간)", ylabel="누적 행 비율")
    axes[2].set(title="사상 길이 vs 참여 관측소 수", xlabel="사상 길이 (시간)", ylabel="관측소 수")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. 타깃과의 관계 (전체 vs 비조석) — 분석용, 모델 변수 금지
# ============================================================
def plot_target_by_length(data, ev, target):
    try:
        reg = pd.read_csv(TIDAL_PATH)
        excl = set(reg.loc[reg["group"].isin(["조석", "고정"]), "station_id"])
    except FileNotFoundError:
        print(f"'{TIDAL_PATH}' 없음 → 비조석 층화 생략")
        excl = set()
    d = data[["event_id", "station_id", target]].copy()
    d["length_h"] = d["event_id"].map(ev["length_h"])
    bins = [0, 3, 6, 12, 24, 48, 200]
    labels = ["1-3h", "4-6h", "7-12h", "13-24h", "25-48h", "49h+"]
    d["구간"] = pd.cut(d["length_h"], bins=bins, labels=labels)
    groups = [("전체", d), (f"비조석 (조석·고정 {len(excl)}곳 제외)", d[~d["station_id"].isin(excl)])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    rhos = []
    for ax, (name, g) in zip(axes, groups):
        s = g.groupby("구간", observed=True)[target]
        stats = pd.DataFrame({"평균": s.mean(), "p95": s.quantile(.95), "RMSE(0예측)": np.sqrt((g[target] ** 2).groupby(g["구간"], observed=True).mean())})
        stats.plot(kind="bar", ax=ax, color=["#2F6DB5", "#8FB3DE", "#9A9A9A"], edgecolor="white", rot=0)
        ax.set_xticklabels([f"{k}\n({n / len(g):.0%})" for k, n in s.size().items()])   # 괄호 = 행 비중
        rho = spearmanr(g["length_h"], g[target]).statistic
        rhos.append(rho)
        ax.set(title=f"{name}\n사상 길이 구간별 타깃 (스피어만 {rho:.2f}, 괄호 = 행 비중)",
               xlabel="사상 길이 구간", ylabel=f"{target} (m)")
    plt.tight_layout()
    plt.show()
    return rhos


# ============================================================
# 6. 진입점 + 해석 요약
# ============================================================
def run_event_length(df, test_df=None, target="target_maxrise_6h"):
    data = add_event_id(df.copy())   # 원본 df는 건드리지 않는다
    sets = [("train", event_lengths(data), C_TR)]
    if test_df is not None:
        sets.append(("test", event_lengths(add_event_id(test_df.copy())), C_TE))

    print("길이 정의: 사상 = 끝 시각 − 첫 시각 + 1h / 관측소-사상 = 그 관측소 행 수\n")
    for name, (ev, se), _ in sets:
        summarize(f"{name} 사상 길이(h)", ev["length_h"])
        summarize(f"{name} 관측소-사상(행)", se["n"])
        short = ev["length_h"] <= 6
        top = ev["length_h"] >= ev["length_h"].quantile(.9)
        print(f"   6h 이하 사상 {short.mean():.0%} (행 비중 {ev.loc[short, 'n_rows'].sum() / ev['n_rows'].sum():.1%}), "
              f"길이 상위 10% 사상의 행 비중 {ev.loc[top, 'n_rows'].sum() / ev['n_rows'].sum():.0%}, "
              f"최장 사상 1개의 행 비중 {ev['n_rows'].max() / ev['n_rows'].sum():.1%}\n")

    plot_event_length(sets)
    plot_station_event(sets)
    ev_tr = sets[0][1][0]
    rhos = plot_target_by_length(data, ev_tr, target) if target in data.columns else [np.nan, np.nan]

    ev_te = sets[1][1][0] if len(sets) > 1 else None
    print("⚠️ 누수 주의: 사상 길이는 사상이 끝나야 알 수 있는 값이야. 모델 변수로 쓰면 안 되고, 분석·fold 설계용으로만 써.")
    print("이 EDA는 전체 데이터 기준이야. 모델링할 땐 train 기준으로 다시 계산해.\n")
    print("── 해석 ──")
    print(f"사상 길이는 오른쪽 꼬리가 긴 분포야. train 중앙값 {ev_tr['length_h'].median():.0f}h인데 최대는 {ev_tr['length_h'].max():.0f}h까지 가.")
    if ev_te is not None:
        print(f"test는 중앙값 {ev_te['length_h'].median():.0f}h, 최대 {ev_te['length_h'].max():.0f}h. 누적분포가 거의 겹쳐서 길이 쪽 분포 차이는 작아. 초장기 사상이 test에 없는 것만 달라.")
    print("개수로는 6h 이하 짧은 사상이 많지만, 행으로 따지면 긴 사상 몇 개가 대부분을 차지해.")
    print(f"최장 사상 하나가 train 행의 {ev_tr['n_rows'].max() / ev_tr['n_rows'].sum():.0%}쯤이라서, 사상 단위 fold를 나눌 때 그 fold만 크기가 튈 수 있어.")
    print("거의 모든 사상에 관측소가 460곳 넘게 들어와. 관측소가 빠지는 건 짧은 사상 몇 개뿐이야.")
    print(f"타깃과 사상 길이의 스피어만은 전체 {rhos[0]:.2f} / 비조석 {rhos[1]:.2f}로 거의 0이야. 그런데 25h 넘는 사상에서 p95랑 RMSE가 확 커져.")
    print("정리하면 사상이 길다고 보통 행이 더 오르는 건 아니고, 큰 상승이 긴 사상에 몰려 있다는 뜻이야. 그래도 사상이 끝나야 아는 값이라 변수로는 못 써.")


run_event_length(df, test_df=test)   # test를 안 불러왔으면 test_df=None
