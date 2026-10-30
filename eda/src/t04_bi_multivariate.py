"""T4. 4장 이변량(그림 4-1~4-6) + 5장 다변량(그림 5-1~5-4, 표 5-1). 전체(회색) / 비조석(파랑) 병기."""
import numpy as np
import pandas as pd
from scipy import stats

from tb_common import (load_tb, station_table, style, figsize, save, save_table, save_json, plt,
                       C_ALL, C_NONTIDAL, C_TIDAL, CAT, SEQ, div_cmap, IMERG, SFC, RADAR, ENV, TARGET, WL)
from common import binned_stats, symlog_x

DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_24h"]
RAIN = IMERG + SFC
GROUPS = [("전체", C_ALL), ("비조석", C_NONTIDAL)]


def sub(df, g):
    return df if g == "전체" else df[~df.is_tidal & ~df.is_stuck]


def binned_panel(ax, df, x, q=20, symlog=True, std=True):
    out = {}
    for g, c in GROUPS:
        d = sub(df, g)
        b = binned_stats(d[x], d[TARGET], q=q)
        ax.plot(b.x_mid, b["mean"], "-o", ms=2.5, color=c, label=f"{g} 평균")
        if std:
            ax.plot(b.x_mid, b["std"], ":", color=c, lw=1.2, label=f"{g} 표준편차")
        out[g] = {"mean_zero_or_low": float(b["mean"].iloc[0]), "mean_top": float(b["mean"].iloc[-1]),
                  "std_low": float(b["std"].iloc[0]), "std_top": float(b["std"].iloc[-1])}
    if symlog:
        symlog_x(ax, 0.1)
    ax.set_title(x, fontsize=8)
    return out


# ---------------- 4장 ----------------
def fig_4_1(tr):
    fig, ax = plt.subplots(2, 3, figsize=figsize(16, 9), sharey=True)
    res = {c: binned_panel(a, tr, c) for a, c in zip(ax.ravel(), RAIN)}
    ax[0, 0].legend(fontsize=6); ax[0, 0].set_ylabel("타깃 (m)"); ax[1, 0].set_ylabel("타깃 (m)")
    for a in ax[1]:
        a.set_xlabel("강우 (symlog, 0은 별도 구간)")
    fig.suptitle("그림 4-1  강우가 많을수록 타깃의 평균과 분산이 함께 커진다 (이분산)", x=0.02, ha="left")
    save(fig, "4-1", "target_vs_rain")
    return res


def fig_4_2(tr):
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 6.5), sharey=True)
    res = {}
    for a, x in zip(ax, ["dwl_1h", "dwl_3h"]):
        for g, c in GROUPS:
            d = sub(tr, g)
            edges = np.unique(np.quantile(d[x], np.linspace(0, 1, 41)))
            k = pd.cut(d[x], edges, include_lowest=True)
            b = d.groupby(k, observed=True).agg(x=(x, "median"), m=(TARGET, "mean"))
            a.plot(b.x, b.m, "-o", ms=2.5, color=c, label=g)
            res[f"{x}_{g}_mean_at_min_bin"] = float(b.m.iloc[0])
            res[f"{x}_{g}_mean_at_max_bin"] = float(b.m.iloc[-1])
        symlog_x(a, 0.01); a.set_xlabel(f"{x} (m, symlog)"); a.axvline(0, color=C_ALL, lw=.6)
    ax[0].set_ylabel("타깃 평균 (m)"); ax[0].legend()
    fig.suptitle("그림 4-2  수위 변화와 타깃: V자의 왼쪽 가지는 조석에서 온다", x=0.02, ha="left")
    save(fig, "4-2", "target_vs_dwl")
    return res


def fig_4_3(tr):
    fig, ax = plt.subplots(1, 3, figsize=figsize(16, 5.5), sharey=True)
    res = {c: binned_panel(a, tr, c, symlog=False) for a, c in zip(ax, RADAR)}
    ax[0].legend(fontsize=6); ax[0].set_ylabel("타깃 (m)")
    fig.suptitle("그림 4-3  레이더 반사도와 타깃", x=0.02, ha="left")
    save(fig, "4-3", "target_vs_radar")
    return res


def fig_4_4(tr, st):
    m = tr[~tr.is_tidal & ~tr.is_stuck].groupby("station_id", observed=True)[TARGET].agg(["mean", "std"])
    d = st.join(m, how="inner")
    fig, ax = plt.subplots(2, 4, figsize=figsize(16, 8.5)); ax = ax.ravel()
    res = {}
    for a, c in zip(ax, ENV):
        x = np.log10(d[c]) if c in ("cat_area", "wsarea") else d[c]
        r = stats.spearmanr(d[c], d["mean"]).statistic
        rs = stats.spearmanr(d[c], d["std"]).statistic
        res[c] = {"rho_mean": float(r), "rho_std": float(rs)}
        a.scatter(x, d["mean"], s=6, color=C_NONTIDAL, alpha=.6)
        a.set_title(f"{c}{' (log10)' if c in ('cat_area', 'wsarea') else ''}\nρ평균 {r:+.2f} · ρ표준편차 {rs:+.2f}", fontsize=7)
    ax[7].axis("off")
    ax[0].set_ylabel("관측소 평균 타깃 (m)"); ax[4].set_ylabel("관측소 평균 타깃 (m)")
    fig.suptitle(f"그림 4-4  유역환경과 관측소 평균 타깃 (비조석, n = {len(d)})", x=0.02, ha="left")
    save(fig, "4-4", "target_vs_env")
    return res


def fig_4_5(tr):
    d = tr[~tr.is_tidal & ~tr.is_stuck]
    m = d.groupby("station_id", observed=True)[TARGET].mean().sort_values()
    pick = list(m.index[:20]) + list(m.index[-20:])
    data = [d.loc[d.station_id == s, TARGET].to_numpy() for s in pick]
    fig, ax = plt.subplots(figsize=figsize(16, 7))
    bp = ax.boxplot(data, whis=(1, 99), showfliers=False, patch_artist=True, widths=.6)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(SEQ[1] if i < 20 else SEQ[4]); b.set_edgecolor("none")
    for k in ("medians",):
        for l in bp[k]:
            l.set_color("black")
    ax.set_xticks(range(1, 41)); ax.set_xticklabels([str(s)[2:] for s in pick], rotation=90, fontsize=6)
    ax.axvline(20.5, color=C_ALL, ls="--", lw=1)
    ax.set_ylabel("타깃 (m), 수염 = 1~99%")
    ax.set_title("그림 4-5  비조석 관측소: 평균 하위 20곳(왼쪽) vs 상위 20곳(오른쪽)")
    save(fig, "4-5", "target_by_station")
    return {"station_mean_p10": float(m.quantile(.1)), "station_mean_p90": float(m.quantile(.9))}


def fig_4_6(tr, st_rel):
    feats = ["wl_t", "wl_rel"] + DWL + RAIN + RADAR + ENV
    rows = {}
    for g, _ in GROUPS:
        d = sub(tr, g).sample(300_000, random_state=0) if len(sub(tr, g)) > 300_000 else sub(tr, g)
        for f in feats:
            rows.setdefault(f, {})[f"{g} Pearson"] = d[f].corr(d[TARGET])
            rows[f][f"{g} Spearman"] = stats.spearmanr(d[f], d[TARGET]).statistic
    t = pd.DataFrame(rows).T.sort_values("비조석 Spearman")
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 10), sharey=True)
    y = np.arange(len(t))
    for a, g in zip(ax, ["전체", "비조석"]):
        a.barh(y + .2, t[f"{g} Spearman"], height=.4, color=C_NONTIDAL, label="Spearman")
        a.barh(y - .2, t[f"{g} Pearson"], height=.4, color=CAT[3], label="Pearson")
        a.set_title(g); a.axvline(0, color=C_ALL, lw=.6)
    ax[0].set_yticks(y); ax[0].set_yticklabels(t.index, fontsize=7); ax[0].legend(loc="lower right")
    ax[0].set_xlabel("타깃과의 상관"); ax[1].set_xlabel("타깃과의 상관")
    fig.suptitle("그림 4-6  Pearson과 Spearman: 차이가 크면 비선형·이상값 영향 (표본 30만 행)", x=0.02, ha="left")
    save(fig, "4-6", "corr_summary")
    save_table(t, "4-1", "corr_with_target")
    return t.round(3).to_dict("index")


# ---------------- 5장 ----------------
def fig_5_1(tr):
    feats = ["wl_rel"] + DWL + RAIN + RADAR
    d = tr[feats].sample(200_000, random_state=0)
    c = d.corr(method="spearman")
    from scipy.cluster.hierarchy import linkage, leaves_list
    order = leaves_list(linkage(1 - c.abs().to_numpy()[np.triu_indices(len(c), 1)], "average"))
    c = c.iloc[order, order]
    fig, ax = plt.subplots(figsize=figsize(16, 13))
    im = ax.imshow(c, cmap=div_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index, fontsize=7)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(c.iloc[i, j]) > .6 else "black")
    fig.colorbar(im, shrink=.7, label="Spearman ρ")
    ax.set_title("그림 5-1  동적 입력 변수 간 상관 (비슷한 변수끼리 정렬, 표본 20만 행)")
    save(fig, "5-1", "input_corr")
    # VIF (log1p 강우, 표준화)
    X = d.copy()
    for r in RAIN:
        X[r] = np.log1p(X[r])
    X = (X - X.mean()) / X.std()
    inv = np.linalg.inv(np.corrcoef(X.to_numpy().T))
    vif = pd.Series(np.diag(inv), index=X.columns).sort_values(ascending=False)
    save_table(vif.to_frame("VIF").round(2), "5-1", "vif")
    pairs = {"imerg_sfc_1h": c.loc["imerg_rain_1h", "sfc_rain_1h"], "imerg_sfc_24h": c.loc["imerg_rain_24h", "sfc_rain_24h"],
             "radar_mean_p95": c.loc["radar_reflectivity_mean", "radar_reflectivity_p95"],
             "radar_mean_ge20": c.loc["radar_reflectivity_mean", "radar_reflectivity_ge20_fraction"]}
    return {"vif": vif.round(2).to_dict(), "pairs": {k: float(v) for k, v in pairs.items()}}


def fig_5_2(st):
    X = st[ENV].astype(float).copy()
    for c in ("cat_area", "wsarea", "mean_elevation", "stream_order"):
        X[c] = np.log(X[c])
    Z = (X - X.mean()) / X.std()
    U, S, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    scores = U * S
    fig, ax = plt.subplots(1, 2, figsize=figsize(16, 7), gridspec_kw={"width_ratios": [1, 1.6]})
    ax[0].bar(range(1, 8), ev * 100, color=C_NONTIDAL); ax[0].plot(range(1, 8), np.cumsum(ev) * 100, "-o", color=CAT[1], ms=3)
    ax[0].set_xlabel("주성분"); ax[0].set_ylabel("설명 분산 (%)"); ax[0].set_title("스크리 그림 (선 = 누적)")
    col = np.where(st.is_tidal, C_TIDAL, C_NONTIDAL)
    ax[1].scatter(scores[:, 0], scores[:, 1], s=6, c=col, alpha=.6)
    k = np.abs(scores[:, :2]).max() / np.abs(Vt[:2]).max() * .8
    for i, c in enumerate(ENV):
        ax[1].arrow(0, 0, Vt[0, i] * k, Vt[1, i] * k, color="black", width=.01, head_width=.12)
        ax[1].text(Vt[0, i] * k * 1.12, Vt[1, i] * k * 1.12, c, fontsize=7)
    ax[1].set_xlabel(f"PC1 ({ev[0]:.0%})"); ax[1].set_ylabel(f"PC2 ({ev[1]:.0%})"); ax[1].set_title("biplot (주황 = 조석)")
    fig.suptitle(f"그림 5-2  유역환경 7변수 주성분분석 (관측소 단위, 면적·표고·차수는 log, n = {len(st)})", x=0.02, ha="left")
    save(fig, "5-2", "env_pca")
    load = pd.DataFrame(Vt[:3].T, index=ENV, columns=["PC1", "PC2", "PC3"])
    save_table(load.round(3), "5-2", "pca_loadings")
    return {"explained": ev.round(4).tolist(), "loadings": load.round(3).to_dict()}


def fig_5_3(tr):
    d = sub(tr, "비조석")
    rb = pd.cut(d.imerg_rain_6h, [-.01, 0, 1, 5, 15, 40, 1e9], labels=["0", "0~1", "1~5", "5~15", "15~40", "40+"])
    envs = [("wsarea", "상류 유역면적"), ("urban_ratio", "도시화율"), ("mean_slope", "평균 경사")]
    fig, ax = plt.subplots(1, 3, figsize=figsize(16, 6), sharey=True)
    res = {}
    for a, (e, lab) in zip(ax, envs):
        stq = d.groupby("station_id", observed=True)[e].first()
        tq = pd.qcut(stq, 3, labels=["하", "중", "상"])
        t = d.station_id.map(tq)
        m = d.groupby([rb, t], observed=True)[TARGET].mean().unstack()
        for (qn, col) in zip(["하", "중", "상"], [SEQ[1], SEQ[3], SEQ[6]]):
            a.plot(range(len(m)), m[qn], "-o", ms=3, color=col, label=f"{lab} {qn}")
        a.set_xticks(range(len(m))); a.set_xticklabels(m.index, fontsize=7)
        a.set_xlabel("IMERG 6h 강우 (mm)"); a.legend(fontsize=6); a.set_title(lab)
        res[e] = m.round(3).to_dict()
    ax[0].set_ylabel("타깃 평균 (m)")
    fig.suptitle("그림 5-3  강우 × 유역환경 상호작용 (비조석, 환경은 관측소 3분위)", x=0.02, ha="left")
    save(fig, "5-3", "interaction")
    return res


def var_decomp(df):
    y = df[TARGET].astype(float); tot = ((y - y.mean()) ** 2).sum()
    st = df.groupby("station_id", observed=True)[TARGET].transform("mean")
    ev = df.groupby("event_block")[TARGET].transform("mean")
    se = df.groupby(["station_id", "event_block"], observed=True)[TARGET].transform("mean")
    s_st = ((st - y.mean()) ** 2).sum() / tot
    s_ev = ((ev - y.mean()) ** 2).sum() / tot
    s_se = ((se - y.mean()) ** 2).sum() / tot - s_st - s_ev
    return {"관측소": s_st, "사상": s_ev, "관측소×사상": max(s_se, 0), "사상 안 시간 변동": 1 - ((se - y.mean()) ** 2).sum() / tot}


def fig_5_4(tr):
    res = {g: var_decomp(sub(tr, g)) for g, _ in GROUPS}
    t = pd.DataFrame(res)
    fig, ax = plt.subplots(figsize=figsize(16, 5))
    left = np.zeros(2)
    for (k, row), col in zip(t.iterrows(), [CAT[0], CAT[3], CAT[2], "#d9d8d2"]):
        ax.barh(range(2), row.values * 100, left=left, color=col, label=k)
        for i, v in enumerate(row.values):
            if v > .04:
                ax.text(left[i] + v * 50, i, f"{v:.0%}", ha="center", va="center", fontsize=8)
        left += row.values * 100
    ax.set_yticks(range(2)); ax.set_yticklabels(t.columns); ax.set_xlabel("타깃 분산 중 비율 (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.35))
    ax.set_title("그림 5-4  타깃 분산 분해: 조석을 빼면 관측소 몫이 줄어든다")
    save(fig, "5-4", "variance_decomp")
    return t.round(4).to_dict()


def main():
    style()
    tr = load_tb("train_sorted", ["event_block", "wl_t", TARGET] + DWL + RAIN + RADAR + ENV)
    g = tr.groupby("station_id", observed=True).wl_t
    tr["wl_rel"] = (tr.wl_t - g.transform("median")) / (g.transform("quantile", .9) - g.transform("quantile", .1) + 1e-3)
    st = station_table(tr)
    res = {"4-1": fig_4_1(tr), "4-2": fig_4_2(tr), "4-3": fig_4_3(tr), "4-4": fig_4_4(tr, st), "4-5": fig_4_5(tr),
           "4-6": fig_4_6(tr, st), "5-1": fig_5_1(tr), "5-2": fig_5_2(st), "5-3": fig_5_3(tr), "5-4": fig_5_4(tr)}
    save_json(res, "t04_bi_multi.json")
    for k in ["4-2", "4-4", "5-1", "5-2", "5-4"]:
        print(k, res[k])
    print("4-6", pd.DataFrame(res["4-6"]).T.tail(8))


if __name__ == "__main__":
    main()
