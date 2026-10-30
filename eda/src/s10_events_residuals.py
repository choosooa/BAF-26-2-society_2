"""S10. 탐색: 강우 원천 시간 정렬, 사상 해부, 오차가 사는 곳, 미래강우의 몫, 빠진 행의 패턴."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from common import (load, FEATURES, TARGET, RESULTS, CACHE, CAT, INK2, MUTED, SEQ, GRID,
                    savefig, save_json, peak_rss, style, div_cmap, rmse)
from s04_hydro import keyed, shift_lookup

style()
RES = {}
DWL = ["dwl_1h", "dwl_3h", "dwl_6h", "dwl_12h", "dwl_24h"]
PAIRS = [("imerg_rain_1h", "sfc_rain_1h", "IMERG(t) vs 지상(t+k)"),
         ("radar_reflectivity_mean", "sfc_rain_1h", "레이더(t) vs 지상(t+k)"),
         ("radar_reflectivity_mean", "imerg_rain_1h", "레이더(t) vs IMERG(t+k)")]
KS = range(-4, 5)


def group_corr(g, x, y, min_n=50):
    d = pd.DataFrame({"g": g, "x": x, "y": y}).dropna()
    d["xy"], d["xx"], d["yy"] = d.x * d.y, d.x ** 2, d.y ** 2
    a = d.groupby("g").agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"), sxy=("xy", "sum"), sxx=("xx", "sum"), syy=("yy", "sum"))
    cov = a.sxy / a.n - a.sx * a.sy / a.n ** 2
    return (cov / np.sqrt((a.sxx / a.n - (a.sx / a.n) ** 2) * (a.syy / a.n - (a.sy / a.n) ** 2))).where(a.n >= min_n)


def alignment(name):
    df = load(f"{name}_sorted", ["station_id", "station_code", "hour", "event_block", "imerg_rain_1h", "sfc_rain_1h",
                                 "radar_reflectivity_mean"])
    key = keyed(df)
    out = {}
    for a, b, lbl in PAIRS:
        xa = df[a].to_numpy(np.float64)
        st = pd.DataFrame({k: group_corr(df.station_code.to_numpy(), xa, shift_lookup(df, key, b, k)) for k in KS})
        ev = pd.DataFrame({k: group_corr(df.event_block.to_numpy(), xa, shift_lookup(df, key, b, k), 200) for k in KS})
        out[lbl] = {"median_curve": st.median().round(3).to_dict(),
                    "station_argmax_share": st.dropna().idxmax(axis=1).value_counts(normalize=True).round(3).sort_index().to_dict(),
                    "event_argmax_share": ev.dropna().idxmax(axis=1).value_counts(normalize=True).round(3).sort_index().to_dict(),
                    "_curve": st.median(), "_q": (st.quantile(.25), st.quantile(.75))}
    # 다음 1시간 지상강수를 누가 가장 잘 '미리' 보나
    nxt = shift_lookup(df, key, "sfc_rain_1h", 1)
    out["next_hour_ground"] = {c: float(group_corr(df.station_code.to_numpy(), df[c].to_numpy(np.float64), nxt).median())
                               for c in ["sfc_rain_1h", "imerg_rain_1h", "radar_reflectivity_mean"]}
    return out


def fig_alignment(tr, te):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (_, _, lbl) in zip(axes, PAIRS):
        for res, col, nm in [(tr, CAT[0], "train"), (te, CAT[1], "test")]:
            c = res[lbl]["_curve"]
            ax.plot(c.index, c.values, color=col, marker="o", ms=4, label=nm)
            if nm == "train":
                lo, hi = res[lbl]["_q"]
                ax.fill_between(c.index, lo.values, hi.values, color=col, alpha=.12, lw=0)
        pk = tr[lbl]["_curve"].idxmax()
        ax.axvline(0, color=MUTED, lw=.8, ls=":")
        ax.axvline(pk, color=CAT[0], lw=1, ls="--")
        sh = tr[lbl]["station_argmax_share"]
        ax.set_title(f"{lbl}\n최대 k={pk:+d} (관측소 {sh.get(pk, 0):.0%}가 이 k에서 최대)", fontsize=10)
        ax.set_xlabel("k (h)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("상관 (관측소별 중앙값, 띠 = IQR)")
    fig.suptitle("순간값인 레이더를 기준으로 삼각측량: 지상은 제시각, IMERG(t)는 '다음 1시간(t→t+1)' 강우와 일치", fontsize=11)
    fig.tight_layout(); savefig(fig, "10a_rain_alignment.png")


def regimes(df, special):
    wet = (df.imerg_rain_6h >= 0.5) | (df.sfc_rain_6h >= 0.5)
    x = df.dwl_1h
    r = np.select([df.station_id.astype(str).isin(special["tidal"]), df.station_id.astype(str).isin(special["daily"]),
                   ~wet & (x.abs() <= 0.01), ~wet & (x < -0.01), ~wet & (x > 0.01),
                   wet & (x.abs() <= 0.01), wet & (x > 0.01), wet & (x < -0.01)],
                  ["조석(강·약)", "24h 조작", "무강우·정지", "무강우·하강", "무강우·상승",
                   "비·아직 반응 없음", "비·상승 중", "비·하강 중"], "기타")
    return pd.Series(r, index=df.index)


def residual_map(df, special):
    y = df[TARGET].to_numpy(np.float64)
    res = {}
    rg = regimes(df, special)
    e_b = (y - df.oof_base) ** 2
    e_n = (y - df.oof_net) ** 2
    tab = pd.DataFrame({"rows": rg.value_counts(normalize=True)})
    tab["sse_base"] = e_b.groupby(rg).sum() / e_b.sum()
    tab["rmse_base"] = np.sqrt(e_b.groupby(rg).mean())
    tab["rmse_net"] = np.sqrt(e_n.groupby(rg).mean())
    tab["bias_base(y-pred)"] = (y - df.oof_base).groupby(rg).mean()
    tab["target_mean"] = pd.Series(y, index=df.index).groupby(rg).mean()
    tab = tab.sort_values("sse_base", ascending=False)
    res["regimes"] = tab.round(4).to_dict("index")

    nt = ~df.station_id.astype(str).isin(special["tidal"] + special["daily"])
    d = df[nt]
    rain = np.maximum(d.imerg_rain_6h, d.sfc_rain_6h)
    rb = pd.cut(rain, [-0.01, 0.5, 2, 5, 10, 20, 50, 1e4], labels=["<0.5", "0.5-2", "2-5", "5-10", "10-20", "20-50", "50+"])
    xb = pd.cut(d.dwl_1h, [-1e3, -0.1, -0.03, -0.01, 0.01, 0.03, 0.1, 0.3, 1e3],
                labels=["<-0.1", "-0.1~-0.03", "-0.03~-0.01", "±0.01", "0.01~0.03", "0.03~0.1", "0.1~0.3", ">0.3"])
    r = (d[TARGET] - d.oof_base)
    bias = r.groupby([xb, rb], observed=False).mean().unstack()
    sse = (r ** 2).groupby([xb, rb], observed=False).sum().unstack() / (r ** 2).sum()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lim = 0.4
    im = axes[0].imshow(bias.to_numpy(dtype=float), cmap=div_cmap().reversed(), vmin=-lim, vmax=lim, aspect="auto", origin="lower")
    axes[0].set_title("평균 잔차 (실제 − 예측): 빨강 = 과소예측", fontsize=10)
    im2 = axes[1].imshow(sse.to_numpy(dtype=float) * 100, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("s", ["#fcfcfb"] + SEQ[1:]),
                         aspect="auto", origin="lower")
    axes[1].set_title("제곱오차 비중 (%)", fontsize=10)
    for ax, M, fmt in [(axes[0], bias, "{:+.2f}"), (axes[1], sse * 100, "{:.1f}")]:
        for (i, j), v in np.ndenumerate(M.to_numpy(dtype=float)):
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color="#0b0b0b")
        ax.set_xticks(range(len(rb.cat.categories))); ax.set_xticklabels(rb.cat.categories)
        ax.set_yticks(range(len(xb.cat.categories))); ax.set_yticklabels(xb.cat.categories)
        ax.set_xlabel("최근 6h 강우 (두 원천 중 큰 값, mm)"); ax.set_ylabel("최근 1h 수위변화 dwl_1h (m)"); ax.grid(False)
    fig.colorbar(im, ax=axes[0], shrink=.8); fig.colorbar(im2, ax=axes[1], shrink=.8)
    fig.suptitle("비조석 관측소에서 기준 모델(HGB)이 틀리는 곳", fontsize=11)
    fig.tight_layout(); savefig(fig, "10c_residual_map.png")
    res["onset_cell"] = {"bias(dwl ±0.01, rain 20-50)": float(bias.loc["±0.01", "20-50"]),
                         "bias(dwl >0.3, rain 50+)": float(bias.loc[">0.3", "50+"])}
    top = (sse.stack() * 100).sort_values(ascending=False).head(6)
    res["top_sse_cells"] = {f"dwl {a} / rain {b}": round(float(v), 2) for (a, b), v in top.items()}
    return res


def events(tr, te, special):
    out = {}
    def ev_stats(d):
        h = d.groupby(["event_block", "hour"]).agg(wet=("imerg_rain_6h", lambda v: (v > 1).mean()), inten=("imerg_rain_1h", "mean"))
        e = h.groupby("event_block").agg(extent=("wet", "mean"), peak=("inten", "max"), hours=("wet", "size"))
        return e
    e_tr, e_te = ev_stats(tr), ev_stats(te)
    y = tr[TARGET].to_numpy(np.float64)
    tr = tr.assign(e0=y ** 2, eb=(y - tr.oof_base) ** 2)
    g = tr.groupby("event_block")
    e_tr["sse0_share"] = g.e0.sum() / tr.e0.sum()
    e_tr["sse_base_share"] = g.eb.sum() / tr.eb.sum()
    e_tr["rmse_base"] = np.sqrt(g.eb.mean())
    e_tr["row_share"] = g.size() / len(tr)
    nt = ~tr.station_id.astype(str).isin(special["tidal"])
    e_tr["sse_base_share_nontidal"] = tr[nt].groupby("event_block").eb.sum() / tr[nt].eb.sum()
    srt = e_tr.sort_values("sse_base_share", ascending=False)
    out["top5_sse_base_share"] = float(srt.sse_base_share.head(5).sum())
    out["top5_row_share"] = float(srt.row_share.head(5).sum())
    out["top10_sse_base_share"] = float(srt.sse_base_share.head(10).sum())
    out["n_events_for_half_sse"] = int((srt.sse_base_share.cumsum() < 0.5).sum() + 1)
    out["top5_events"] = srt.head(5)[["hours", "extent", "peak", "row_share", "sse_base_share", "rmse_base"]].round(3).to_dict("index")
    out["corr_event_rmse_vs"] = {c: float(e_tr.rmse_base.corr(e_tr[c], method="spearman")) for c in ["extent", "peak", "hours"]}
    out["test_vs_train"] = {c: {"train_p50": float(e_tr[c].median()), "test_p50": float(e_te[c].median()),
                                "train_max": float(e_tr[c].max()), "test_max": float(e_te[c].max())} for c in ["extent", "peak", "hours"]}
    out["test_events_above_train_p90_peak"] = int((e_te.peak > e_tr.peak.quantile(.9)).sum())
    e_tr.to_csv(RESULTS / "s10_events_train.csv"); e_te.to_csv(RESULTS / "s10_events_test.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    top = srt.head(5).index
    rest = e_tr.index.difference(top)
    ax.scatter(e_tr.loc[rest, "extent"], e_tr.loc[rest, "peak"], s=20 + 3000 * e_tr.loc[rest, "sse_base_share"],
               color="#b4b2a9", alpha=.7, lw=0, label="train 사상 (크기 = 오차 비중)")
    ax.scatter(e_tr.loc[top, "extent"], e_tr.loc[top, "peak"], s=20 + 3000 * e_tr.loc[top, "sse_base_share"],
               color=CAT[1], alpha=.85, lw=0, label="오차 상위 5개 사상")
    for b in top:
        ax.annotate(f"#{b} ({e_tr.sse_base_share[b]:.0%})", (e_tr.extent[b], e_tr.peak[b]), fontsize=8, color=INK2,
                    xytext=(6, 4), textcoords="offset points")
    ax.scatter(e_te.extent, e_te.peak, s=34, facecolor="none", edgecolor=CAT[0], lw=1.3, label="test 사상 (입력만)")
    ax.set_xlabel("강우 공간 범위 (6h 강우>1mm 관측소 비율, 사상 평균)"); ax.set_ylabel("최대 시간 강도 (전 관측소 평균 IMERG 1h 최대, mm)")
    ax.set_title("사상 해부: 넓게 오래 온 비가 오차를 지배한다", fontsize=10); ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    cs = srt.sse_base_share.cumsum().to_numpy()
    ax.plot(np.arange(1, len(cs) + 1), cs * 100, color=CAT[0], lw=2)
    ax.plot(np.arange(1, len(cs) + 1), srt.row_share.cumsum().to_numpy() * 100, color=MUTED, lw=1.5, ls="--")
    ax.text(len(cs) * .55, cs[len(cs) // 2] * 100 + 3, "오차 누적", color=INK2, fontsize=9)
    ax.text(len(cs) * .55, srt.row_share.cumsum().to_numpy()[len(cs) // 2] * 100 - 8, "행 누적", color=INK2, fontsize=9)
    ax.set_xlabel("사상 (오차 큰 순)"); ax.set_ylabel("누적 비중 (%)")
    ax.set_title(f"122개 중 {out['n_events_for_half_sse']}개 사상이 오차의 절반", fontsize=10)
    fig.tight_layout(); savefig(fig, "10b_event_anatomy.png")
    return out


def future_rain(df, special):
    key = keyed(df)
    out = {}
    fut = {c: np.zeros(len(df)) for c in ["sfc", "imerg"]}
    n_ok = np.zeros(len(df), int)
    fmax = np.zeros(len(df))
    for j in range(1, 7):
        s = shift_lookup(df, key, "sfc_rain_1h", j)
        i = shift_lookup(df, key, "imerg_rain_1h", j)
        ok = np.isfinite(s)
        n_ok += ok
        fut["sfc"] += np.nan_to_num(s); fut["imerg"] += np.nan_to_num(i)
        fmax = np.maximum(fmax, np.nan_to_num(s))
    full = n_ok == 6
    df = df.assign(fut_sfc6=np.where(full, fut["sfc"], np.nan), fut_imerg6=np.where(full, fut["imerg"], np.nan),
                   fut_sfc_max=np.where(full, fmax, np.nan))
    d = df[full]
    out["rows_with_full_future"] = float(full.mean())
    out["spearman_fut_sfc6_with"] = {c: float(d[c].corr(d.fut_sfc6, method="spearman")) for c in
                                     ["sfc_rain_1h", "sfc_rain_6h", "imerg_rain_1h", "imerg_rain_6h", "radar_reflectivity_mean",
                                      "radar_reflectivity_ge20_fraction"]}
    dry_now = (d.sfc_rain_1h == 0) & (d.imerg_rain_1h == 0)
    out["dry_now_share"] = float(dry_now.mean())
    out["dry_now_fut_rain_share(>1mm)"] = float((d.fut_sfc6[dry_now] > 1).mean())
    out["dry_now_spearman_radar_vs_fut"] = float(d.radar_reflectivity_mean[dry_now].corr(d.fut_sfc6[dry_now], method="spearman"))
    # 오라클: 미래 6h 강우를 알았다면 오차가 얼마나 줄까 (진단용 — 모델 변수로 쓰는 것 아님)
    y = d[TARGET].to_numpy(np.float64)
    base = FEATURES + DWL
    oracle = base + ["fut_sfc6", "fut_imerg6", "fut_sfc_max"]
    pred = {"base": np.zeros(len(d)), "oracle": np.zeros(len(d))}
    for tri, vai in GroupKFold(5).split(d, groups=d.event_block):
        tr, va = d.iloc[tri], d.iloc[vai]
        st = tr.groupby("station_code")[TARGET].agg(st_mean="mean", st_std="std")
        tr, va = tr.join(st, on="station_code"), va.join(st, on="station_code")
        for nm, feats in [("base", base + ["st_mean", "st_std"]), ("oracle", oracle + ["st_mean", "st_std"])]:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50, random_state=0)
            m.fit(tr[feats], tr[TARGET]); pred[nm][vai] = m.predict(va[feats])
    tid = d.station_id.astype(str).isin(special["tidal"]).to_numpy()
    out["oracle"] = {"rmse_base": rmse(y, pred["base"]), "rmse_oracle": rmse(y, pred["oracle"]),
                     "mse_reduction": float(1 - np.mean((y - pred["oracle"]) ** 2) / np.mean((y - pred["base"]) ** 2)),
                     "nontidal_mse_reduction": float(1 - np.mean((y[~tid] - pred["oracle"][~tid]) ** 2) / np.mean((y[~tid] - pred["base"][~tid]) ** 2))}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    rb = pd.qcut(d.radar_reflectivity_mean[dry_now].rank(method="first"), 10, labels=False)
    g = d[dry_now].groupby(rb.to_numpy()).agg(radar=("radar_reflectivity_mean", "median"), fut=("fut_sfc6", "mean"),
                                              p_rain=("fut_sfc6", lambda v: (v > 1).mean()))
    ax.bar(np.arange(len(g)), g.p_rain * 100, color=SEQ[3], width=.7)
    for i, (r, p) in enumerate(zip(g.radar, g.p_rain)):
        ax.text(i, p * 100 + 0.5, f"{p:.0%}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(g))); ax.set_xticklabels([f"{v:.1f}" for v in g.radar], fontsize=8)
    ax.set_xlabel("지금 비가 0일 때의 레이더 반사도 (10분위 중앙값)"); ax.set_ylabel("향후 6h 지상강수 > 1mm 확률 (%)")
    ax.set_title("지금은 맑아도 레이더가 다가오는 비를 먼저 본다", fontsize=10)
    ax = axes[1]
    labels = ["기준 모델\n(현재 입력)", "오라클\n(+미래 6h 강우)"]
    vals = [out["oracle"]["rmse_base"], out["oracle"]["rmse_oracle"]]
    ax.bar([0, 1], vals, color=[MUTED, CAT[0]], width=.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=10, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels); ax.set_ylabel("RMSE (m, 미래 6행이 있는 행)")
    ax.set_title(f"미래 강우를 완벽히 안다면: MSE {out['oracle']['mse_reduction']:.0%} 감소 (상한선)", fontsize=10)
    fig.tight_layout(); savefig(fig, "10d_future_rain.png")
    RES["radar_nowcast_bins"] = g.round(3).to_dict("index")
    return out


def gaps(tr, special):
    out = {}
    bl = pd.read_csv(RESULTS / "s01_event_blocks.csv")
    bl = bl[bl.split == "train"]
    present = tr.groupby(["event_block", "station_code"]).size().rename("n").reset_index()
    present = present.merge(bl[["event_block", "n_hours", "h0"]], on="event_block")
    grid_n = int(present.n_hours.sum()); rows = int(present.n.sum())
    out["missing_cells_in_present_blocks"] = float(1 - rows / grid_n)
    per_hour = tr.groupby("hour").station_code.nunique()
    blk = tr.groupby("hour").event_block.first()
    st_in_block = present.groupby("event_block").station_code.nunique()
    frac = 1 - per_hour / blk.map(st_in_block)
    out["hours_with_>20pct_missing"] = int((frac > 0.2).sum())
    out["hour_missing_frac_quantiles"] = frac.quantile([.5, .9, .99, 1]).round(3).to_dict()
    reg = tr.groupby("hour").imerg_rain_1h.mean()
    dec = pd.qcut(reg.rank(method="first"), 10, labels=False)
    out["missing_frac_by_regional_rain_decile"] = frac.groupby(dec).mean().round(4).to_dict()
    key = keyed(tr)
    nxt = shift_lookup(tr, key, "hour", 1)
    before_gap = np.isnan(nxt) & (tr.hour.to_numpy() < tr.event_block.map(bl.set_index("event_block").h0 + bl.set_index("event_block").n_hours - 1).to_numpy())
    y = tr[TARGET]
    out["rows_before_internal_gap"] = int(before_gap.sum())
    out["before_gap_vs_all"] = {"target_mean": [float(y[before_gap].mean()), float(y.mean())],
                                "abs_dwl1_mean": [float(tr.dwl_1h[before_gap].abs().mean()), float(tr.dwl_1h.abs().mean())],
                                "sfc6_mean": [float(tr.sfc_rain_6h[before_gap].mean()), float(tr.sfc_rain_6h.mean())]}
    # 공통 결측(같은 시각에 여러 관측소 동시 누락) vs 개별 결측
    return out


def main():
    special_j = json.load(open(RESULTS / "s08_fingerprint.json", encoding="utf-8"))
    special = {"tidal": special_j["periodic"]["strong_tidal"] + special_j["periodic"]["weak_tidal"],
               "daily": special_j["periodic"]["daily_only"] + special_j["resolution"]["never_changes"]}
    a_tr, a_te = alignment("train"), alignment("test")
    fig_alignment(a_tr, a_te)
    RES["alignment"] = {"train": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v
                                  for k, v in a_tr.items()},
                        "test": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v
                                 for k, v in a_te.items()}}
    print(json.dumps(RES["alignment"], ensure_ascii=False, indent=1)[:3500], flush=True)

    cols = ["station_id", "station_code", "hour", "event_block", "pos_in_block", TARGET] + FEATURES + DWL
    tr = load("train_sorted", cols)
    oof = pd.read_parquet(CACHE / "oof_s09.parquet")
    tr = tr.merge(oof, on=["station_code", "hour"], how="left")
    te = load("test_sorted", ["station_id", "station_code", "hour", "event_block", "imerg_rain_1h", "imerg_rain_6h"])
    RES["residuals"] = residual_map(tr, special)
    RES["events"] = events(tr, te, special)
    RES["gaps"] = gaps(tr, special)
    RES["future_rain"] = future_rain(tr, special)
    save_json(RES, "s10_events_residuals.json")
    print(json.dumps({k: v for k, v in RES.items() if k != "alignment"}, ensure_ascii=False, indent=1, default=str)[:9000])
    peak_rss("s10")


if __name__ == "__main__":
    main()
