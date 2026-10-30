"""S4. 수문 분석: 수문곡선, 강우→수위 지체시간, 조석(주기성) 관측소 탐지."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load, TARGET, savefig, save_json, peak_rss

RES = {}


def keyed(df):
    return df["station_code"].to_numpy(np.int64) * 1_000_000 + df["hour"].to_numpy(np.int64)


def shift_lookup(df, key, col, dh):
    s = pd.Series(df[col].to_numpy(np.float64), index=key)
    return s.reindex(key + dh).to_numpy()


def station_corr(df, x, y):
    """관측소별 Pearson 상관 (NaN 제외), 벡터화."""
    d = pd.DataFrame({"s": df["station_id"].to_numpy(), "x": x, "y": y}).dropna()
    d["xy"], d["xx"], d["yy"] = d.x * d.y, d.x * d.x, d.y * d.y
    g = d.groupby("s", observed=True).agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
                                          sxy=("xy", "sum"), sxx=("xx", "sum"), syy=("yy", "sum"))
    cov = g.sxy / g.n - g.sx / g.n * g.sy / g.n
    vx = g.sxx / g.n - (g.sx / g.n) ** 2
    vy = g.syy / g.n - (g.sy / g.n) ** 2
    return (cov / np.sqrt(vx * vy)).where(g.n > 50)


def periodicity(df, key):
    """dwl_1h 자기상관 lag 1..26 → 조석(반일주조 ~12.4h) 신호."""
    acf = {}
    x = df["dwl_1h"].to_numpy(np.float64)
    for L in range(1, 27):
        acf[L] = station_corr(df, x, shift_lookup(df, key, "dwl_1h", -L))
    A = pd.DataFrame(acf)
    tidal_score = A[[12, 13]].max(axis=1) - A[[6, 7]].min(axis=1)  # 12~13h 양의 상관 + 6h 음의 상관
    return A, tidal_score


def lag_analysis(df, key):
    dwl = df["dwl_1h"].to_numpy(np.float64)
    out = {}
    for src in ["imerg_rain_1h", "sfc_rain_1h"]:
        cc = {L: station_corr(df, shift_lookup(df, key, src, -L) if L else df[src].to_numpy(np.float64), dwl) for L in range(0, 25)}
        C = pd.DataFrame(cc)
        out[src] = C
    return out


def main():
    cols = ["station_id", "station_code", "hour", "event_block", "pos_in_block", "wl_t", "dwl_1h", TARGET,
            "imerg_rain_1h", "sfc_rain_1h", "wsarea", "cat_area", "mean_slope", "urban_ratio"]
    df = load("train_sorted", cols)
    key = keyed(df)
    st = df.groupby("station_id", observed=True).agg(wsarea=("wsarea", "first"), tmean=(TARGET, "mean"),
                                                     tstd=(TARGET, "std"), sse=(TARGET, lambda s: float((s.astype(float) ** 2).sum())))
    st["sse_share"] = st.sse / st.sse.sum()

    # --- 주기성(조석) ---
    A, tscore = periodicity(df, key)
    st["tidal_score"] = tscore
    st["acf12"] = A[[12, 13]].max(axis=1)
    st["acf6"] = A[[6, 7]].min(axis=1)
    tidal = st[(st.acf12 > 0.6) & (st.acf6 < 0)].sort_values("sse_share", ascending=False)
    top_sse = st.sort_values("sse_share", ascending=False).head(10)
    RES["tidal"] = {"rule": "acf(dwl_1h) max(12,13h) > 0.6 and min(6,7h) < 0",
                    "n_tidal": int(len(tidal)), "tidal_sse_share(0pred)": float(tidal.sse_share.sum()),
                    "tidal_row_share": float(df.station_id.isin(tidal.index).mean()),
                    "tidal_stations": tidal[["tmean", "tstd", "sse_share", "acf12", "acf6"]].round(3).to_dict("index"),
                    "top10_sse": top_sse[["tmean", "tstd", "sse_share", "acf12", "acf6"]].round(3).to_dict("index"),
                    "top10_sse_that_are_tidal": int(top_sse.index.isin(tidal.index).sum())}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for s_, row in A.iterrows():
        is_t = s_ in tidal.index
        axes[0].plot(A.columns, row.values, color="#D85A30" if is_t else "#B4B2A9", alpha=.9 if is_t else .15, lw=1.2 if is_t else .6)
    axes[0].axhline(0, color="k", lw=.5); axes[0].set_xlabel("lag (h)"); axes[0].set_ylabel("ACF of dwl_1h")
    axes[0].set_title(f"관측소별 수위변화 자기상관 (주황 = 조석 의심 {len(tidal)}개)")
    axes[1].scatter(st.acf12, st.sse_share * 100, s=10, c=np.where(st.index.isin(tidal.index), "#D85A30", "#534AB7"))
    axes[1].set_yscale("log"); axes[1].set_xlabel("ACF 12~13h (반일주기 신호)"); axes[1].set_ylabel("0 예측 SSE 기여 (%)")
    axes[1].set_title("주기성 vs RMSE 기여"); axes[1].grid(alpha=.3)
    fig.tight_layout(); savefig(fig, "04a_tidal_detection.png")

    # 조석 관측소 예시 시계열 (가장 긴 train 사상)
    long_block = df.groupby("event_block").hour.nunique().idxmax()
    ex = list(tidal.index[:3]) + list(st[~st.index.isin(tidal.index)].sort_values("sse_share", ascending=False).index[:1])
    fig, axes = plt.subplots(len(ex), 1, figsize=(12, 2.2 * len(ex)), sharex=True)
    for ax, s_ in zip(np.atleast_1d(axes), ex):
        d = df[(df.station_id == s_) & (df.event_block == long_block)].sort_values("hour")
        ax.plot(d.pos_in_block, d.wl_t, color="#D85A30" if s_ in tidal.index else "#534AB7")
        ax2 = ax.twinx(); ax2.bar(d.pos_in_block, d.imerg_rain_1h, color="#1D9E75", alpha=.4, width=1)
        ax2.invert_yaxis(); ax2.set_ylabel("강우", fontsize=8)
        ax.set_ylabel("wl_t"); ax.set_title(f"{s_} ({'조석 의심' if s_ in tidal.index else '비조석, SSE 상위'})", fontsize=10)
    np.atleast_1d(axes)[-1].set_xlabel(f"사상 {long_block} 시작 후 시간 (h)")
    fig.tight_layout(); savefig(fig, "04b_tidal_examples.png")

    # --- 수문곡선: 타깃 평균이 큰 사상, 유역면적 3분위별 2개 관측소 (조석 제외) ---
    nt = df[~df.station_id.isin(tidal.index)]
    ev = nt.groupby("event_block").agg(m=(TARGET, "mean"), h=("hour", "nunique"))
    b = ev[ev.h >= 30].m.idxmax()
    d = nt[nt.event_block == b]
    smax = d.groupby("station_id", observed=True).agg(mx=(TARGET, "max"), ws=("wsarea", "first"))
    smax["tert"] = pd.qcut(smax.ws.rank(method="first"), 3, labels=["소", "중", "대"])
    pick = smax.sort_values("mx", ascending=False).groupby("tert", observed=True).head(2).sort_values(["tert", "mx"])
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for ax, (s_, r) in zip(axes.flat, pick.iterrows()):
        dd = d[d.station_id == s_].sort_values("hour")
        ax.plot(dd.pos_in_block, dd.wl_t - dd.wl_t.iloc[0], color="#534AB7", label="wl_t (시작 대비)")
        ax2 = ax.twinx(); ax2.bar(dd.pos_in_block, dd.imerg_rain_1h, color="#1D9E75", alpha=.5, width=1, label="IMERG 1h")
        ax2.bar(dd.pos_in_block, dd.sfc_rain_1h, color="#0F6E56", alpha=.3, width=.5, label="지상 1h")
        ax2.invert_yaxis(); ax2.set_ylim(max(dd.imerg_rain_1h.max(), dd.sfc_rain_1h.max(), 1) * 2.5, 0)
        ax.set_title(f"{s_}  유역면적 {r.tert} (wsarea={r.ws:.1f})", fontsize=10); ax.set_ylabel("Δ수위 (m)")
    axes[0][0].legend(loc="center left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(f"사상 {b} 시작 후 시간 (h)")
    fig.suptitle("수문곡선 예시: 강우(위, 뒤집힘) → 수위 반응(아래)"); fig.tight_layout(); savefig(fig, "04c_hydrographs.png")
    RES["hydrograph_event"] = int(b)

    # --- 지체시간 ---
    nt_mask = ~df.station_id.isin(tidal.index)
    C = lag_analysis(df, key)
    lag = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for src, colr in [("imerg_rain_1h", "#1D9E75"), ("sfc_rain_1h", "#0F6E56")]:
        Cs = C[src].loc[C[src].index.isin(st.index[~st.index.isin(tidal.index)])]
        peak = Cs.idxmax(axis=1)
        st[f"lag_{src}"] = peak
        st[f"lagcorr_{src}"] = Cs.max(axis=1)
        axes[0].plot(Cs.columns, Cs.median(), "o-", ms=3, color=colr, label=f"{src} (중앙값)")
        axes[1].hist(peak.dropna(), bins=np.arange(-0.5, 25), alpha=.6, color=colr, label=src)
        ok = st[f"lag_{src}"].notna()
        lag[src] = {"peak_lag_quantiles": peak.quantile([.1, .25, .5, .75, .9]).to_dict(),
                    "median_peak_corr": float(Cs.max(axis=1).median()),
                    "spearman_lag_vs_wsarea": float(st.loc[ok, f"lag_{src}"].corr(st.loc[ok, "wsarea"], method="spearman"))}
    axes[0].set_xlabel("강우 선행 시간 L (h)"); axes[0].set_ylabel("corr(rain(t−L), Δwl(t))"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title("강우→수위변화 교차상관 (비조석 관측소)")
    axes[1].set_xlabel("상관이 최대인 지연 (h)"); axes[1].set_title("관측소별 지체시간 분포"); axes[1].legend()
    ok = st.lag_imerg_rain_1h.notna()
    axes[2].scatter(st.wsarea[ok], st.lag_imerg_rain_1h[ok] + np.random.default_rng(0).uniform(-.2, .2, ok.sum()), s=8, alpha=.6, color="#534AB7")
    axes[2].set_xscale("log"); axes[2].set_xlabel("wsarea (log)"); axes[2].set_ylabel("지체시간 (h, IMERG)"); axes[2].set_title("유역면적 vs 지체시간")
    fig.tight_layout(); savefig(fig, "04d_lag_time.png")
    RES["lag"] = lag
    st.drop(columns="sse").to_csv(__import__("common").RESULTS / "s04_station_summary.csv")
    save_json(RES, "s04_hydro.json")
    import json
    r = dict(RES); r["tidal"] = {k: v for k, v in RES["tidal"].items() if k not in ("tidal_stations",)}
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    peak_rss("s04")


if __name__ == "__main__":
    main()
