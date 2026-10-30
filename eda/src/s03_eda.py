"""S3. 일반 EDA: 타깃, 입력 분포, 관계(binned), 강우 3종 비교, 이분산성, 유역 조절효과, 관측소 이질성."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import symlog_x, load, IMERG, SFC, RADAR, ENV, WL, TARGET, savefig, save_json, peak_rss, binned_stats

C_TR, C_TE, C_ACC = "#534AB7", "#D85A30", "#1D9E75"
RES = {}


def target_dist(df):
    y = df[TARGET].to_numpy(np.float64)
    q = np.quantile(y, [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1])
    sq = np.sort(y ** 2)[::-1]
    top = {f"top{p}pct_share_of_SSE(0pred)": float(sq[: int(len(sq) * p / 100)].sum() / sq.sum()) for p in (0.1, 1, 5)}
    RES["target"] = {"quantiles": dict(zip(["min", "p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9", "max"], q.round(4))),
                     "mean": float(y.mean()), "std": float(y.std()), "skew": float(pd.Series(y).skew()),
                     "abs_gt_1m": float((np.abs(y) > 1).mean()), **top}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(y, bins=200, color=C_TR); axes[0].set_yscale("log"); axes[0].set_title("타깃 분포 (y 로그)")
    axes[0].set_xlabel("target_maxrise_6h (m)")
    axes[1].hist(y[(y > -0.3) & (y < 0.5)], bins=160, color=C_TR); axes[1].set_title("중심부 확대 (−0.3~0.5m)")
    cum = np.cumsum(sq) / sq.sum()
    xs = np.arange(1, len(sq) + 1) / len(sq) * 100
    axes[2].plot(xs, cum, color=C_TE); axes[2].set_xscale("log"); axes[2].set_xlabel("상위 행 비율 (%)")
    axes[2].set_ylabel("0 예측 시 제곱오차 누적 비중"); axes[2].set_title("극값이 RMSE를 지배하는 정도"); axes[2].grid(alpha=.3)
    fig.tight_layout(); savefig(fig, "03a_target_dist.png")


def input_dists(df):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, c in zip(axes[0], ["imerg_rain_1h", "imerg_rain_6h", "sfc_rain_1h", "sfc_rain_6h"]):
        v = df[c].to_numpy()
        nz = v[v > 0]
        ax.hist(np.log10(nz), bins=80, color=C_ACC)
        ax.set_title(f"{c}\n0 비율 {np.mean(v == 0):.1%}, 비영 log10 분포", fontsize=10)
    for ax, c in zip(axes[1][:3], RADAR):
        ax.hist(df[c], bins=100, color="#888780"); ax.set_title(c, fontsize=10); ax.set_yscale("log")
    ax = axes[1][3]
    for c in ["dwl_1h", "dwl_6h"]:
        v = df[c].clip(-1, 1)
        ax.hist(v, bins=200, alpha=.6, label=c)
    ax.set_yscale("log"); ax.legend(); ax.set_title("수위 변화량 (±1m 절단)", fontsize=10)
    fig.tight_layout(); savefig(fig, "03b_input_dists.png")


def env_station(df):
    env = df.groupby("station_id", observed=True)[ENV].first()
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, c in zip(axes.flat, ENV):
        ax.hist(env[c], bins=40, color="#993C1D", alpha=.8); ax.set_title(c, fontsize=10)
    sns.heatmap(env.corr(method="spearman"), ax=axes.flat[7], annot=True, fmt=".1f", cmap="RdBu_r", vmin=-1, vmax=1,
                cbar=False, annot_kws={"size": 7}); axes.flat[7].set_title("유역환경 Spearman (관측소 단위)", fontsize=10)
    fig.tight_layout(); savefig(fig, "03c_env_station.png")
    RES["env_skew"] = env.skew().round(2).to_dict()


def band(ax, g, color, label=None):
    ax.plot(g["x_mid"], g["mean"], "o-", color=color, ms=3, label=label)
    ax.fill_between(g["x_mid"], g["q10"], g["q90"], color=color, alpha=.15)


def rel_rain(df):
    y = df[TARGET]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    out = {}
    for row, cols, col in [(0, IMERG, C_ACC), (1, SFC, "#0F6E56")]:
        for j, c in enumerate(cols):
            g = binned_stats(df[c], y, q=20)
            ax = axes[row][j]; band(ax, g, col)
            symlog_x(ax); ax.set_title(c, fontsize=10); ax.grid(alpha=.3)
            out[c] = {"mean_at_zero": float(g.loc["0", "mean"]) if "0" in g.index else None,
                      "mean_top_bin": float(g["mean"].iloc[-1]), "std_zero": float(g.loc["0", "std"]) if "0" in g.index else None,
                      "std_top_bin": float(g["std"].iloc[-1])}
    axes[0][0].set_ylabel("타깃 평균 (띠: 10~90%)"); axes[1][0].set_ylabel("타깃 평균 (띠: 10~90%)")
    fig.suptitle("강우 구간별 타깃 (0은 별도 구간, x는 symlog)"); fig.tight_layout(); savefig(fig, "03d_target_vs_rain.png")
    RES["target_vs_rain"] = out

    # 이분산성
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, colr in [("imerg_rain_6h", C_ACC), ("sfc_rain_6h", "#0F6E56"), ("radar_reflectivity_p95", "#888780")]:
        g = binned_stats(df[c], y, q=20)
        ax.plot(np.arange(len(g)), g["std"], "o-", ms=3, color=colr, label=c)
    ax.set_xlabel("변수 분위 구간 (0 → 큼)"); ax.set_ylabel("타깃 표준편차 (m)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("이분산성: 비가 많을수록 타깃 분산이 커지는가"); savefig(fig, "03e_heteroscedasticity.png")


def rel_wl(df):
    y = df[TARGET]
    # 관측소 내 표준화 wl_t
    g = df.groupby("station_id", observed=True)["wl_t"]
    z = (df["wl_t"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df["wl_t_z_station"] = z.astype(np.float32)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    out = {}
    for ax, c in zip(axes, ["dwl_1h", "dwl_6h", "dwl_24h", "wl_t_z_station"]):
        x = df[c]
        ok = x.notna()
        gb = binned_stats(x[ok].clip(x.quantile(.001), x.quantile(.999)), y[ok], q=30)
        band(ax, gb, C_TR); ax.set_title(c, fontsize=10); ax.grid(alpha=.3); ax.axhline(0, color="k", lw=.5)
        out[c] = float(pd.Series(x[ok]).corr(y[ok], method="spearman"))
    axes[0].set_ylabel("타깃 평균 (띠: 10~90%)")
    fig.suptitle("수위 추세·상대수위 vs 타깃"); fig.tight_layout(); savefig(fig, "03f_target_vs_wl.png")
    RES["spearman_wl_features"] = out


def corr_matrix(df):
    s = df.sample(200_000, random_state=0)
    cols = ["wl_t", "dwl_1h", "dwl_3h", "dwl_6h", "dwl_24h", "wl_t_z_station"] + IMERG + SFC + RADAR + ENV + [TARGET]
    c = s[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(c, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6.5}, ax=ax, cbar_kws={"shrink": .6})
    ax.set_title("Spearman 상관 (표본 200k)"); savefig(fig, "03g_spearman.png")
    RES["spearman_with_target"] = c[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False).round(3).to_dict()


def rain_sources(df):
    out = {}
    for h in ["1h", "6h", "24h"]:
        a, b = df[f"imerg_rain_{h}"], df[f"sfc_rain_{h}"]
        out[h] = {"spearman": float(a.corr(b, method="spearman")), "pearson": float(a.corr(b)),
                  "imerg>0_sfc=0": float(((a > 0) & (b == 0)).mean()), "imerg=0_sfc>0": float(((a == 0) & (b > 0)).mean()),
                  "both>0": float(((a > 0) & (b > 0)).mean())}
        both = (a > 0.5) & (b > 0.5)
        out[h]["ratio_sfc_over_imerg_median(both>0.5)"] = float((b[both] / a[both]).median())
    both = (df.imerg_rain_24h > 1) & (df.sfc_rain_24h > 1)
    r = np.log10(df.sfc_rain_24h[both] / df.imerg_rain_24h[both])
    out["radar_vs_imerg1h_spearman"] = float(df.radar_reflectivity_mean.corr(df.imerg_rain_1h, method="spearman"))
    out["radar_vs_sfc1h_spearman"] = float(df.radar_reflectivity_mean.corr(df.sfc_rain_1h, method="spearman"))
    RES["rain_sources"] = out
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    s = df[both].sample(min(both.sum(), 150_000), random_state=0)
    axes[0].hexbin(np.log10(s.imerg_rain_24h), np.log10(s.sfc_rain_24h), gridsize=60, bins="log", cmap="viridis")
    axes[0].plot([0, 3], [0, 3], "r--", lw=1); axes[0].set_xlabel("log10 imerg_rain_24h (mm)"); axes[0].set_ylabel("log10 sfc_rain_24h (단위?)")
    axes[0].set_title("IMERG vs 지상 (둘 다 >1)")
    axes[1].hist(r, bins=100, color=C_ACC); axes[1].axvline(0, color="r", ls="--", lw=1)
    axes[1].set_title(f"log10(sfc/imerg) 24h, 중앙값 비 {10 ** r.median():.2f}"); axes[1].set_xlabel("log10 비율")
    g = binned_stats(df.radar_reflectivity_mean, df.imerg_rain_1h, q=25)
    axes[2].plot(g.x_mid, g["mean"], "o-", ms=3, color="#888780", label="imerg_rain_1h 평균")
    g2 = binned_stats(df.radar_reflectivity_mean, df.sfc_rain_1h, q=25)
    axes[2].plot(g2.x_mid, g2["mean"], "o-", ms=3, color="#0F6E56", label="sfc_rain_1h 평균")
    axes[2].set_xlabel("radar_reflectivity_mean 구간"); axes[2].legend(); axes[2].set_title("레이더 반사도 vs 1h 강우"); axes[2].grid(alpha=.3)
    fig.tight_layout(); savefig(fig, "03h_rain_sources.png")


def env_moderation(df):
    y = df[TARGET]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    out = {}
    for ax, e in zip(axes, ["wsarea", "mean_slope", "urban_ratio"]):
        st = df.groupby("station_id", observed=True)[e].first()
        tert = pd.qcut(st.rank(method="first"), 3, labels=["하", "중", "상"])
        t = df["station_id"].map(tert).astype(str)
        out[e] = {}
        for lab, colr in [("하", "#B4B2A9"), ("중", "#7F77DD"), ("상", "#26215C")]:
            m = (t == lab).to_numpy()
            g = binned_stats(df.imerg_rain_6h[m], y[m], q=12)
            ax.plot(g.x_mid, g["mean"], "o-", ms=3, color=colr, label=f"{e} {lab}")
            out[e][lab] = {"mean_top_bin": float(g["mean"].iloc[-1]), "mean_zero": float(g.loc["0", "mean"]) if "0" in g.index else None}
        symlog_x(ax); ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_xlabel("imerg_rain_6h (mm)")
        ax.set_title(f"{e} 3분위별 강우→타깃")
    axes[0].set_ylabel("타깃 평균 (m)")
    fig.tight_layout(); savefig(fig, "03i_env_moderation.png")
    RES["env_moderation"] = out


def station_hetero(df):
    y = df[TARGET].astype(np.float64)
    tot = y.var()
    st = df.groupby("station_id", observed=True)[TARGET].agg(["mean", "std", "max", "min", "size"])
    ev = df.groupby("event_block")[TARGET].mean()
    se = df.groupby(["station_id", "event_block"], observed=True)[TARGET].transform("mean")
    RES["variance_decomp"] = {
        "total_var": float(tot),
        "between_station_share": float(df.groupby("station_id", observed=True)[TARGET].transform("mean").var() / tot),
        "between_event_share": float(df.groupby("event_block")[TARGET].transform("mean").var() / tot),
        "between_station_x_event_share": float(se.var() / tot),
        "station_std_quantiles": st["std"].quantile([.1, .5, .9, .99]).round(4).to_dict(),
        "station_max_quantiles": st["max"].quantile([.1, .5, .9, .99]).round(3).to_dict(),
        "top10_station_share_of_SSE(0pred)": float(df.assign(sq=y ** 2).groupby("station_id", observed=True)["sq"].sum()
                                                   .sort_values(ascending=False).head(10).sum() / (y ** 2).sum()),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(st["std"], bins=60, color=C_TR); axes[0].set_xscale("log"); axes[0].set_title("관측소별 타깃 표준편차 (log x)")
    axes[1].scatter(st["mean"], st["std"], s=8, alpha=.6, color=C_TR); axes[1].set_xlabel("관측소 평균"); axes[1].set_ylabel("관측소 표준편차")
    axes[1].set_yscale("log"); axes[1].set_title("관측소 평균 vs 변동성")
    axes[2].hist(ev, bins=40, color=C_TE); axes[2].set_title("사상별 타깃 평균 (122개)")
    fig.tight_layout(); savefig(fig, "03j_station_event_hetero.png")

    # 사상 내 위치에 따른 타깃
    p = df.groupby(df.pos_in_block.clip(upper=48))[TARGET].agg(["mean", "std", "size"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(p.index, p["mean"], "o-", ms=3, color=C_TR, label="평균"); ax2 = ax.twinx()
    ax2.plot(p.index, p["std"], "s--", ms=3, color=C_TE, label="표준편차")
    ax.set_xlabel("사상 시작 후 경과 시간 (h, 48 이상 묶음)"); ax.set_ylabel("타깃 평균"); ax2.set_ylabel("타깃 표준편차")
    ax.set_title("사상 내 위치별 타깃"); ax.grid(alpha=.3); savefig(fig, "03k_target_by_event_pos.png")
    RES["target_by_pos_first6h_mean"] = p["mean"].head(6).round(4).to_dict()


def main():
    cols = ["station_id", "event_block", "pos_in_block", "wl_t", "dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h", TARGET] \
        + IMERG + SFC + RADAR + ENV
    df = load("train_sorted", cols)
    target_dist(df); input_dists(df); env_station(df)
    rel_rain(df); rel_wl(df); corr_matrix(df); rain_sources(df); env_moderation(df); station_hetero(df)
    save_json(RES, "s03_eda.json")
    import json
    print(json.dumps(RES, ensure_ascii=False, indent=1, default=str))
    peak_rss("s03")


if __name__ == "__main__":
    main()
