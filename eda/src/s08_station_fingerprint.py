"""S8. 탐색: 관측소 행동 지문(fingerprint) → 비지도 유형화 + 데이터 품질 이상 발굴.

가설 없이 관측소마다 '수위가 어떻게 움직이는가'를 수치 지문으로 만들고, 군집이 스스로 드러내는 유형을 본다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer

from common import (load, ENV, IMERG, SFC, RADAR, TARGET, RESULTS, CAT, INK2, MUTED, SEQ, GRID as GRID_LINE,
                    savefig, save_json, peak_rss, style, div_cmap)
from s04_hydro import keyed, shift_lookup, station_corr

style()
RES = {}
DF_CACHE = {}
BEHAV = ["acf1", "acf12", "acf24", "dry_sd_log", "rain_rho", "tstd_log", "flat_share", "neg_share", "asym"]
BEHAV_KO = {"acf1": "변화 지속성(ACF1)", "acf12": "12h 주기(ACF12)", "acf24": "24h 주기(ACF24)",
            "dry_sd_log": "무강우 변동(log)", "rain_rho": "강우 반응(ρ)", "tstd_log": "타깃 변동(log)",
            "flat_share": "무변화 비율", "neg_share": "하강 타깃 비율", "asym": "급상승/급하강 비대칭"}


def grp(v, s, how):
    return getattr(pd.Series(v).groupby(s, observed=True), how)()


def fingerprints(df, key):
    s = df["station_id"].astype(str).to_numpy()
    x = df["dwl_1h"].to_numpy(np.float64)
    y = df[TARGET].to_numpy(np.float64)
    fp = pd.DataFrame({"n": grp(np.ones(len(df)), s, "sum")})
    df = df.assign(_s=s)
    df["station_id"] = s  # station_corr 그룹 키를 문자열로 통일
    for L in (1, 6, 12, 24):
        fp[f"acf{L}"] = station_corr(df, x, shift_lookup(df, key, "dwl_1h", -L))
    dry = ((df.imerg_rain_24h < 0.5) & (df.sfc_rain_24h < 0.5)).to_numpy()
    wet = ((df.imerg_rain_6h >= 10) | (df.sfc_rain_6h >= 10)).to_numpy()
    fp["dry_share"] = grp(dry.astype(float), s, "mean")
    fp["dry_sd"] = grp(np.where(dry, x, np.nan), s, "std")
    fp["dry_sd_log"] = np.log10(fp.dry_sd + 1e-3)
    med = grp(x, s, "median")
    fp["mad"] = grp(np.abs(x - med.reindex(s).to_numpy()), s, "median")
    fp["flat_share"] = grp((x == 0).astype(float), s, "mean")
    fp["zero_target"] = grp((y == 0).astype(float), s, "mean")
    fp["neg_share"] = grp((y < 0).astype(float), s, "mean")
    fp["tstd"] = grp(y, s, "std"); fp["tstd_log"] = np.log10(fp.tstd + 1e-3)
    fp["tmax"] = grp(y, s, "max")
    fp["tp99"] = pd.Series(y).groupby(s).quantile(.99)
    rain6 = np.maximum(df.imerg_rain_6h, df.sfc_rain_6h)
    rr = pd.Series(rain6.to_numpy()).groupby(s).rank(pct=True).to_numpy()
    yr = pd.Series(y).groupby(s).rank(pct=True).to_numpy()
    fp["rain_rho"] = station_corr(df, rr, yr)
    fp["wet_gain"] = grp(np.where(wet, y, np.nan), s, "mean") - grp(np.where(dry, y, np.nan), s, "mean")
    fp.loc[grp(wet.astype(float), s, "sum") < 20, "wet_gain"] = np.nan

    # 상승/하강 비대칭: 가장 빠른 상승(q99) vs 가장 빠른 하강(|q01|). 자연하천=빨리 오르고 천천히 내림, 조석=대칭
    q99 = pd.Series(x).groupby(s).quantile(.99)
    q01 = pd.Series(x).groupby(s).quantile(.01).abs()
    fp["asym"] = np.log10((q99 + 0.01) / (q01 + 0.01))

    # 스파이크(튀었다 바로 복귀) / 계단(튀고 유지) — 무강우 구간만
    nxt = shift_lookup(df, key, "dwl_1h", 1)
    thr = np.maximum(5 * 1.4826 * fp["mad"].reindex(s).to_numpy(), 0.05)
    big = (np.abs(x) > thr) & dry
    spike = big & (np.sign(nxt) == -np.sign(x)) & (np.abs(nxt) > 0.7 * np.abs(x))
    step = big & ~spike & np.isfinite(nxt)
    fp["spike_rate"] = grp(spike.astype(float), s, "mean")
    fp["step_rate"] = grp(step.astype(float), s, "mean")

    # 계측 분해능: 0이 아닌 최소 |Δ수위|, 값이 놓인 격자(cm / 10cm)
    ax_ = np.abs(x)
    fp["min_step"] = grp(np.where(ax_ > 1e-6, ax_, np.nan), s, "min").round(4)
    w = df.wl_t.to_numpy(np.float64)
    fp["cm_grid"] = grp((np.abs(w * 100 - np.round(w * 100)) < 2e-3).astype(float), s, "mean")
    fp["dm_grid"] = grp((np.abs(w * 10 - np.round(w * 10)) < 1e-3).astype(float), s, "mean")

    # 레이더 무강우 바탕값(클러터 의심)
    dry_strict = dry & (df.imerg_rain_1h == 0).to_numpy() & (df.sfc_rain_1h == 0).to_numpy()
    fp["radar_dry_med"] = grp(np.where(dry_strict, df.radar_reflectivity_mean, np.nan), s, "median")
    fp["radar_dry_ge20"] = grp(np.where(dry_strict, df.radar_reflectivity_ge20_fraction, np.nan), s, "mean")
    fp["radar_wet_med"] = grp(np.where(wet, df.radar_reflectivity_mean, np.nan), s, "median")

    # 원천 간 편향: log10(지상/IMERG) 24h, 둘 다 > 1
    both = ((df.imerg_rain_24h > 1) & (df.sfc_rain_24h > 1)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(df.sfc_rain_24h.to_numpy(np.float64) / df.imerg_rain_24h.to_numpy(np.float64))
    fp["bias_sfc_imerg"] = grp(np.where(both, lr, np.nan), s, "median")
    fp.loc[grp(both.astype(float), s, "sum") < 30, "bias_sfc_imerg"] = np.nan

    # 결측(사상 안에서 빠진 시각) 비율
    bl = pd.read_csv(RESULTS / "s01_event_blocks.csv")
    blocks_hours = bl.set_index("event_block").n_hours
    sb = df.groupby(["_s", "event_block"]).size().rename("n").reset_index()
    sb["h"] = sb.event_block.map(blocks_hours)
    agg = sb.groupby("_s")[["n", "h"]].sum()
    fp["miss_in_present_blocks"] = 1 - agg.n / agg.h
    fp["miss_all"] = 1 - fp.n / int(bl.loc[bl.split == "train", "n_hours"].sum())

    env = df.groupby("_s")[ENV].first()
    fp = fp.join(env)
    fp["sse_share"] = grp(y ** 2, s, "sum") / (y ** 2).sum()
    tid = pd.read_csv(RESULTS / "s04_station_summary.csv", index_col=0)
    fp["tidal"] = ((tid.acf12 > 0.6) & (tid.acf6 < 0)).reindex(fp.index).fillna(False).astype(bool)
    return fp


def top_values(df):
    """프로파일링 관점: 한 값이 비정상적으로 자주 나오는가(기본값·채움값 의심)."""
    out = {}
    for c in ["wl_t", "imerg_rain_1h", "sfc_rain_1h", "radar_reflectivity_mean", "radar_reflectivity_p95",
              "radar_reflectivity_ge20_fraction", TARGET]:
        vc = df[c].round(4).value_counts(normalize=True).head(5)
        out[c] = {str(k): round(float(v), 4) for k, v in vc.items()}
    # 0이 아닌 값 중 최빈값 (0 제외 기본값 탐지)
    for c in ["radar_reflectivity_mean", "radar_reflectivity_p95", "sfc_rain_1h", "imerg_rain_1h"]:
        v = df[c][df[c] != 0].round(4).value_counts(normalize=True).head(3)
        out[c + "_nonzero"] = {str(k): round(float(q), 4) for k, q in v.items()}
    # 지상강수 값 격자 (0.5 단위?)
    v = df.sfc_rain_1h[df.sfc_rain_1h > 0].to_numpy(np.float64)
    out["sfc_1h_on_0.5_grid"] = float((np.abs(v * 2 - np.round(v * 2)) < 1e-3).mean())
    out["sfc_1h_on_0.1_grid"] = float((np.abs(v * 10 - np.round(v * 10)) < 1e-3).mean())
    return out


def cluster(fp):
    X = fp[BEHAV].copy()
    X = X.fillna(X.median())
    Q = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit_transform(X)
    sil, labs = {}, {}
    for k in range(3, 9):
        km = KMeans(k, n_init=30, random_state=0).fit(Q)
        sil[k], labs[k] = float(silhouette_score(Q, km.labels_)), km.labels_
    k = max(range(4, 8), key=lambda kk: sil[kk])  # 해석 가능한 입도(4~7) 안에서 실루엣 최대
    lab = pd.Series(labs[k], index=fp.index)
    order = fp.groupby(lab)["tstd"].median().sort_values().index  # 변동성 오름차순으로 번호 고정
    lab = lab.map({old: i + 1 for i, old in enumerate(order)})
    pcs = PCA(2, random_state=0).fit(Q)
    P = pcs.transform(Q)
    RES["cluster"] = {"silhouette": sil, "k": k, "pca_var": pcs.explained_variance_ratio_.round(3).tolist(),
                      "pca_loadings": pd.DataFrame(pcs.components_.T, index=BEHAV, columns=["PC1", "PC2"]).round(2).to_dict()}
    return lab, P, Q


def describe_clusters(fp, lab):
    fp = fp.assign(cl=lab)
    prof = fp.groupby("cl").agg(n=("n", "size"), rows=("n", "sum"), sse_share=("sse_share", "sum"),
                                tidal=("tidal", "sum"), **{c: (c, "median") for c in BEHAV + ["tstd", "tmax", "wet_gain",
                                                                                          "dry_sd", "wsarea", "urban_ratio",
                                                                                          "mean_slope", "mean_elevation"]})
    prof["row_share"] = prof.rows / prof.rows.sum()
    # 유역환경만으로 유형을 맞힐 수 있는가
    Xe = np.log1p(fp[ENV].clip(lower=0))
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = cross_val_score(rf, Xe, lab, cv=cv, scoring="accuracy")
    bacc = cross_val_score(rf, Xe, lab, cv=cv, scoring="balanced_accuracy")
    # 같은 소권역 값을 공유하는 관측소끼리 같은 유형인가
    key = fp[ENV].round(4).astype(str).agg("|".join, axis=1)
    same, tot = 0, 0
    for _, g in fp.assign(k=key).groupby("k"):
        if len(g) > 1:
            ls = g.cl.to_numpy()
            for i in range(len(ls)):
                for j in range(i + 1, len(ls)):
                    tot += 1; same += ls[i] == ls[j]
    p = lab.value_counts(normalize=True)
    RES["env_explains_type"] = {"rf_accuracy": float(acc.mean()), "rf_balanced_accuracy": float(bacc.mean()),
                                "majority_baseline": float(p.max()), "chance_balanced": 1 / len(p),
                                "same_subbasin_pairs": tot, "same_subbasin_same_type": float(same / tot) if tot else None,
                                "random_pair_same_type": float((p ** 2).sum())}
    return prof


def fig_clusters(fp, lab, P, prof):
    k = lab.nunique()
    fig, axes = plt.subplots(1, k, figsize=(3.1 * k, 3.2), sharex=True, sharey=True)
    for i, ax in enumerate(axes, start=1):
        m = (lab == i).to_numpy()
        ax.scatter(P[~m, 0], P[~m, 1], s=7, color="#d9d8d3", lw=0)
        ax.scatter(P[m, 0], P[m, 1], s=12, color=CAT[0], lw=0)
        t = fp.tidal.to_numpy() & m
        ax.scatter(P[t, 0], P[t, 1], s=26, facecolor="none", edgecolor=CAT[1], lw=1.2)
        r = prof.loc[i]
        ax.set_title(f"C{i}  관측소 {int(r.n)}곳\n행 {r.row_share:.0%} · 0예측 SSE {r.sse_share:.0%}", fontsize=9.5)
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.suptitle("관측소 행동 지문의 비지도 군집 (파랑 = 해당 유형, 주황 테두리 = 조석)", fontsize=11)
    fig.tight_layout(); savefig(fig, "08a_station_types.png")

    cols = BEHAV
    Z = (prof[cols] - fp[cols].median()) / (fp[cols].quantile(.75) - fp[cols].quantile(.25)).replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 0.55 * k + 1.6))
    im = ax.imshow(Z.clip(-2.5, 2.5).to_numpy(), cmap=div_cmap(), vmin=-2.5, vmax=2.5, aspect="auto")
    for (i, j), v in np.ndenumerate(prof[cols].to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="#0b0b0b")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([BEHAV_KO[c] for c in cols], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(k)); ax.set_yticklabels([f"C{i}" for i in prof.index])
    ax.grid(False); fig.colorbar(im, ax=ax, shrink=.8, label="전체 관측소 대비 (IQR 단위)")
    ax.set_title("유형별 지문 중앙값 (칸 안 숫자 = 원값, 색 = 전체 대비 높고 낮음)")
    savefig(fig, "08b_type_profile.png")


def fig_signatures(df, key, lab):
    """유형별 '움직임 서명': 수위변화 자기상관, 강우→수위변화 교차상관 (관측소 중앙값)."""
    d = df.assign(station_id=df.station_id.astype(str))
    x = d.dwl_1h.to_numpy(np.float64)
    acf = pd.DataFrame({L: station_corr(d, x, shift_lookup(d, key, "dwl_1h", -L)) for L in range(1, 27)})
    xc = pd.DataFrame({L: station_corr(d, shift_lookup(d, key, "sfc_rain_1h", -L) if L else d.sfc_rain_1h.to_numpy(np.float64), x)
                       for L in range(0, 13)})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    out = {}
    for i in sorted(lab.unique()):
        m = lab.index[lab == i]
        a, c = acf.loc[m].median(), xc.loc[m].median()
        col = CAT[i - 1]
        axes[0].plot(a.index, a.values, color=col, lw=2)
        axes[0].text(a.index[-1] + 0.4, a.values[-1], f"C{i}", color=INK2, fontsize=9, va="center")
        axes[1].plot(c.index, c.values, color=col, lw=2, marker="o", ms=4)
        axes[1].text(c.index[-1] + 0.2, c.values[-1], f"C{i}", color=INK2, fontsize=9, va="center")
        out[f"C{i}"] = {"acf_12": float(a[12]), "acf_24": float(a[24]), "xc_peak_lag": int(c.idxmax()), "xc_peak": float(c.max())}
    axes[0].axhline(0, color=MUTED, lw=.8); axes[0].set_xlabel("시차 (h)"); axes[0].set_ylabel("수위변화 자기상관 (중앙값)")
    axes[0].set_title("유형별 리듬: 12·24시간 주기가 보이는 유형은?")
    axes[1].axhline(0, color=MUTED, lw=.8); axes[1].set_xlabel("지상강수 선행 시간 L (h)"); axes[1].set_ylabel("corr(강수(t−L), Δ수위(t))")
    axes[1].set_title("유형별 강우 반응: 얼마나 세게, 얼마나 빨리")
    for ax in axes:
        ax.legend(handles=[plt.Line2D([], [], color=CAT[i - 1], lw=2, label=f"C{i}") for i in sorted(lab.unique())], fontsize=8)
    fig.tight_layout(); savefig(fig, "08c_type_signatures.png")
    RES["type_signatures"] = out


def periodic_classes(fp):
    strong = fp.tidal
    weak = (~strong) & fp.acf12.between(0.35, 0.6) & (fp.acf6 < -0.2)
    daily = (~strong) & (fp.acf24 >= 0.4) & (fp.acf12 < 0.15)
    return strong, weak, daily


def fig_quality(fp):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axes[0][0]
    v = fp.radar_dry_med.dropna()
    ax.hist(v, bins=50, color=SEQ[3])
    thr = float(v.median() + 3 * (v.quantile(.75) - v.quantile(.25)))
    ax.axvline(thr, color=CAT[1], lw=1.2, ls="--")
    ax.set_title(f"맑은 날 레이더 반사도 (관측소별 중앙값)\n점선 오른쪽 {int((v > thr).sum())}곳: 비 없이도 높음 → 클러터 의심")
    ax.set_xlabel("radar_reflectivity_mean (무강우 행)"); ax.set_ylabel("관측소 수")
    RES["radar_clutter"] = {"threshold": thr, "n_high": int((v > thr).sum()),
                            "stations": v[v > thr].sort_values(ascending=False).round(2).head(15).to_dict(),
                            "dry_median_all": float(v.median()), "wet_median_all": float(fp.radar_wet_med.median())}
    ax = axes[0][1]
    strong, weak, daily = periodic_classes(fp)
    other = ~(strong | weak | daily)
    ax.scatter(fp.acf12[other], fp.acf24[other], s=8, color="#d9d8d3", lw=0, label=f"기타 {int(other.sum())}")
    for m, col, lb in [(strong, CAT[1], "강한 조석"), (weak, CAT[2], "약한 조석"), (daily, CAT[0], "24h 주기만")]:
        ax.scatter(fp.acf12[m], fp.acf24[m], s=26, color=col, lw=0, label=f"{lb} {int(m.sum())}")
    for sid in fp.index[daily | (fp.flat_share >= 0.999)]:
        ax.annotate(sid, (fp.acf12[sid], fp.acf24[sid]), fontsize=7.5, color=INK2, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("수위변화 자기상관 12h"); ax.set_ylabel("수위변화 자기상관 24h"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("주기성 지도: 조석은 연속 스펙트럼, 24h 주기는 따로")
    RES["periodic"] = {"strong_tidal": fp.index[strong].tolist(), "weak_tidal": fp.index[weak].tolist(),
                       "daily_only": fp.index[daily].tolist(),
                       "daily_only_rain_rho": fp.loc[daily, "rain_rho"].round(3).to_dict(),
                       "weak_tidal_sse_share": float(fp.loc[weak, "sse_share"].sum())}
    ms = fp.min_step.fillna(-1)
    RES["resolution"] = {"stations_min_step_1cm": int((ms.between(0.0095, 0.0105)).sum()),
                         "never_changes": fp.index[fp.flat_share >= 0.999].tolist(),
                         "flat_share_quantiles": fp.flat_share.quantile([.1, .5, .9, .99]).round(3).to_dict()}

    ax = axes[1][0]
    rate = (fp.spike_rate + fp.step_rate).sort_values(ascending=False).head(15)
    ax.barh(range(len(rate))[::-1], fp.loc[rate.index, "spike_rate"] * 100, color=CAT[1], label="스파이크(튀고 복귀)")
    ax.barh(range(len(rate))[::-1], fp.loc[rate.index, "step_rate"] * 100, left=fp.loc[rate.index, "spike_rate"] * 100,
            color=SEQ[3], label="계단(튀고 유지)")
    ax.set_yticks(range(len(rate))[::-1]); ax.set_yticklabels([f"{s}{' (조석)' if fp.tidal[s] else ''}" for s in rate.index], fontsize=8)
    ax.set_xlabel("무강우 행 중 비율 (%)"); ax.legend(fontsize=8); ax.set_title("비 없이 급변: 센서 튐 / 수문·보 조작 의심")
    RES["jumps"] = {"spike_rate_all": float(np.average(fp.spike_rate, weights=fp.n)),
                    "step_rate_all": float(np.average(fp.step_rate, weights=fp.n)),
                    "top": (fp.loc[rate.index, ["spike_rate", "step_rate", "tidal"]]).round(4).astype(str).to_dict("index")}
    ax = axes[1][1]
    ok = fp.bias_sfc_imerg.notna()
    rho = {c: float(fp.loc[ok, "bias_sfc_imerg"].corr(fp.loc[ok, c], method="spearman"))
           for c in ["mean_elevation", "mean_slope", "forest_ratio", "urban_ratio", "wsarea"]}
    RES["rain_bias"] = {"spearman_vs_env": rho, "median": float(fp.bias_sfc_imerg.median()),
                        "q10_q90": fp.bias_sfc_imerg.quantile([.1, .9]).round(3).tolist()}
    ex = [s_ for s_ in RES["periodic"]["daily_only"]][:2]
    ax.set_title("24h 주기 관측소: 비와 무관하게 매일 같은 패턴")
    RES["daily_example_block"] = None
    if ex:
        wl = DF_CACHE["wl"]
        bl = pd.read_csv(RESULTS / "s01_event_blocks.csv")
        b = bl[bl.split == "train"].sort_values("n_hours", ascending=False).iloc[0].event_block
        for j, sid in enumerate(ex):
            dd = wl[(wl.station_id == sid) & (wl.event_block == b)].sort_values("pos_in_block")
            ax.plot(dd.pos_in_block, dd.wl_t - dd.wl_t.median(), color=CAT[j], lw=1.4, label=f"{sid} (강우 반응 ρ={fp.rain_rho[sid]:.2f})")
        for h in range(0, int(wl.pos_in_block.max()) + 1, 24):
            ax.axvline(h, color=GRID_LINE, lw=.8, ls=":")
        ax.legend(fontsize=8); ax.set_xlabel(f"사상 {b} 시작 후 시간 (h, 점선 = 24h 간격)"); ax.set_ylabel("수위 − 중앙값 (m)")
        RES["daily_example_block"] = int(b)
    fig.tight_layout(); savefig(fig, "08d_data_quirks.png")


def main():
    cols = ["station_id", "station_code", "hour", "event_block", "pos_in_block", "wl_t", "dwl_1h", TARGET] + IMERG + SFC + RADAR + ENV
    df = load("train_sorted", cols)
    key = keyed(df)
    RES["top_values"] = top_values(df)
    fp = fingerprints(df, key)
    lab, P, Q = cluster(fp)
    prof = describe_clusters(fp, lab)
    fp["type"] = lab
    fp.to_csv(RESULTS / "s08_station_fingerprints.csv")
    prof.to_csv(RESULTS / "s08_type_profile.csv")
    RES["type_profile"] = prof.round(3).to_dict("index")
    fig_clusters(fp, lab, P, prof)
    fig_signatures(df, key, lab)
    DF_CACHE["wl"] = df[["station_id", "event_block", "pos_in_block", "wl_t"]].assign(station_id=df.station_id.astype(str))
    fig_quality(fp)
    save_json(RES, "s08_fingerprint.json")
    import json
    print(json.dumps({k: v for k, v in RES.items() if k != "type_profile"}, ensure_ascii=False, indent=1, default=str)[:7000])
    print(prof.round(3).T.to_string())
    peak_rss("s08")


if __name__ == "__main__":
    main()
