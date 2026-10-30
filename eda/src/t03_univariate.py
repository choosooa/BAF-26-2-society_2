"""T3. 3장 단변량: 그림 3-1~3-6, 표 3-1."""
import numpy as np
import pandas as pd
from scipy import stats

from tb_common import (load_tb, station_table, style, figsize, save, save_table, save_json, plt,
                       C_ALL, C_NONTIDAL, C_TIDAL, CAT, IMERG, SFC, RADAR, ENV, TARGET, TIDAL)
from common import symlog_x

DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_24h"]


def fig_target(tr):
    y = tr[TARGET].to_numpy(float)
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 6.5))
    e = np.logspace(np.log10(0.015), 1, 20)  # 1cm 계단보다 넓은 구간
    bins = np.concatenate([-e[::-1], [-0.005, 0.005], e])
    ax[0].hist(y, bins=bins, color=C_NONTIDAL)
    symlog_x(ax[0], 0.01); ax[0].set_yscale("log")
    ax[0].set_xlabel("타깃 (m, symlog)"); ax[0].set_ylabel("행 수 (log)")
    ax[0].set_title("분포: 0 근처에 몰리고 양쪽 꼬리가 길다")
    rng = np.random.default_rng(0)
    s = np.sort(rng.choice(y, 20000, replace=False))
    q = stats.norm.ppf((np.arange(1, len(s) + 1) - .5) / len(s))
    ax[1].plot(q, s, ".", ms=2, color=C_NONTIDAL)
    ax[1].plot(q, y.mean() + y.std() * q, color=C_ALL, lw=1, ls="--", label="정규분포라면")
    ax[1].set_xlabel("정규분포 기준 위치 (σ, 표준편차 배수)"); ax[1].set_ylabel("6시간 내 최대 상승 (m)"); ax[1].legend()
    ax[1].set_title("Q-Q plot: 정규보다 훨씬 두꺼운 꼬리")
    fig.suptitle(f"그림 3-1  타깃 분포 (n = {len(y):,})", x=0.02, ha="left")
    save(fig, "3-1", "target_dist")
    sq = np.sort(y ** 2)[::-1]
    return {"mean": y.mean(), "median": float(np.median(y)), "std": y.std(), "skew": float(stats.skew(y)),
            "kurtosis": float(stats.kurtosis(y)), "pos": float((y > 0).mean()), "zero": float((y == 0).mean()),
            "neg": float((y < 0).mean()), "p99": float(np.quantile(y, .99)), "max": float(y.max()), "min": float(y.min()),
            "top1pct_sse_share": float(sq[: len(sq) // 100].sum() / sq.sum())}


def fig_station(tr):
    g = tr.groupby("station_id", observed=True).agg(mean=(TARGET, "mean"), std=(TARGET, "std"),
                                                    sse=(TARGET, lambda s: (s ** 2).sum()), tidal=("is_tidal", "first"))
    g = g[g["std"] > 0]
    fig, ax = plt.subplots(figsize=figsize(16, 8))
    for flag, c, lab in [(False, C_NONTIDAL, "기타 관측소"), (True, C_TIDAL, "조석 관측소 15곳 (6.3에서 확인)")]:
        d = g[g.tidal == flag]
        ax.scatter(d["std"], d["mean"], s=12 if not flag else 28, color=c, alpha=.7 if not flag else 1, label=f"{lab} n={len(d)}")
    ax.set_xscale("log"); ax.set_xlabel("관측소별 타깃 표준편차 (m, log)"); ax.set_ylabel("관측소별 타깃 평균 (m)")
    ax.legend(loc="upper left")
    ax.set_title("그림 3-2  꼬리는 어디서 오나: 변동이 유독 큰 관측소 무리")
    save(fig, "3-2", "station_mean_std")
    top_std = g.sort_values("std", ascending=False).head(15)
    return {"tidal_in_top15_std": int(top_std.tidal.sum()),
            "tidal_sse_share_zero_pred": float(g.loc[g.tidal, "sse"].sum() / g["sse"].sum()),
            "tidal_row_share": float(tr.is_tidal.mean()),
            "std_median_tidal": float(g.loc[g.tidal, "std"].median()), "std_median_other": float(g.loc[~g.tidal, "std"].median())}


def fig_wl(tr):
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 6.5))
    ax[0].hist(tr.wl_t, bins=200, color=C_ALL); ax[0].set_yscale("log")
    ax[0].set_xlabel("wl_t (m)"); ax[0].set_ylabel("행 수 (log)"); ax[0].set_title("절대수위: 기준면 차이로 여러 봉우리")
    e = np.logspace(np.log10(0.015), 1, 20)
    bins = np.concatenate([-e[::-1], [-0.005, 0.005], e])
    for c, col in zip(DWL, CAT):
        h, e = np.histogram(tr[c], bins=bins)
        ax[1].step(e[1:], h, where="pre", color=col, label=c, lw=1.2)
    symlog_x(ax[1], 0.01); ax[1].set_yscale("log"); ax[1].legend()
    ax[1].set_xlabel("수위 차분 (m, symlog)"); ax[1].set_title("차분: 0 중심, 창이 길수록 넓다")
    fig.suptitle("그림 3-3  수위는 절대값이 아니라 변화량으로 본다", x=0.02, ha="left")
    save(fig, "3-3", "wl_vs_dwl")


def fig_rain(tr):
    fig, ax = plt.subplots(1, 3, figsize=figsize(16, 6), sharey=True)
    for a, w in zip(ax, ["1h", "6h", "24h"]):
        for c, col, lab in [(f"imerg_rain_{w}", CAT[0], "IMERG"), (f"sfc_rain_{w}", CAT[1], "지상")]:
            x = tr[c]; z = (x == 0).mean()
            a.hist(np.log10(x[x > 0]), bins=60, histtype="step", color=col, lw=1.3, label=f"{lab} (0: {z:.0%})")
        a.set_title(w); a.set_xlabel("log10 강우 (양수만)"); a.legend(fontsize=7)
    ax[0].set_ylabel("행 수")
    fig.suptitle("그림 3-4  강우는 0이 많고, 양수 부분은 로그 척도에서 넓게 퍼진다", x=0.02, ha="left")
    save(fig, "3-4", "rain_dist")


def fig_radar(tr):
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 6))
    for c, col in zip(RADAR[:2], CAT):
        ax[0].hist(tr[c], bins=100, histtype="step", color=col, lw=1.3, label=c.replace("radar_reflectivity_", ""))
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_xlabel("반사도 (단위 미기재)"); ax[0].set_title("평균·p95")
    f = tr.radar_reflectivity_ge20_fraction
    ax[1].hist(f, bins=50, color=C_NONTIDAL); ax[1].set_yscale("log")
    ax[1].set_xlabel("ge20 비율"); ax[1].set_title(f"ge20 비율: 정확히 0 또는 1 = {((f == 0) | (f == 1)).mean():.0%}")
    fig.suptitle("그림 3-5  레이더 반사도", x=0.02, ha="left")
    save(fig, "3-5", "radar_dist")
    return {"ge20_zero_or_one": float(((f == 0) | (f == 1)).mean()), "ge20_zero": float((f == 0).mean())}


def fig_env(st):
    fig, ax = plt.subplots(2, 4, figsize=figsize(16, 8.5)); ax = ax.ravel()
    for a, c in zip(ax, ENV):
        x = st[c].astype(float)
        a.hist(x, bins=30, color=C_NONTIDAL)
        a.set_title(f"{c}  왜도 {stats.skew(x):.1f}", fontsize=8)
    ax[7].hist(np.log1p(st.cat_area), bins=30, color=CAT[2]); ax[7].set_title(f"log1p(cat_area)  왜도 {stats.skew(np.log1p(st.cat_area)):.1f}", fontsize=8)
    fig.suptitle(f"그림 3-6  유역환경 (관측소 단위, n = {len(st)})", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "3-6", "env_station")


def table_3_1(tr, st):
    rows = {}
    for c in ["wl_t"] + DWL + IMERG + SFC + RADAR + [TARGET] + ENV:
        x = (st[c] if c in ENV else tr[c]).astype(float)
        sk = stats.skew(x)
        rec = ("log1p" if (x.min() >= 0 and sk > 2) else
               "차분 사용" if c == "wl_t" else "없음(트리) / 강건 척도" if abs(sk) > 2 else "없음")
        rows[c] = {"단위": "관측소" if c in ENV else "행", "평균": x.mean(), "중앙값": x.median(), "표준편차": x.std(),
                   "IQR": x.quantile(.75) - x.quantile(.25), "왜도": sk, "첨도": stats.kurtosis(x),
                   "0 비율": (x == 0).mean(), "권장 변환": rec}
    t = pd.DataFrame(rows).T
    save_table(t, "3-1", "summary")
    return t


def main():
    style()
    tr = load_tb("train_sorted", ["wl_t", TARGET] + DWL + IMERG + SFC + RADAR + ENV)
    st = station_table(tr)
    res = {"target": fig_target(tr), "station": fig_station(tr)}
    fig_wl(tr); fig_rain(tr)
    res["radar"] = fig_radar(tr)
    fig_env(st)
    t = table_3_1(tr, st)
    save_json(res, "t03_univariate.json")
    print(res); print(t[["왜도", "0 비율", "권장 변환"]])


if __name__ == "__main__":
    main()
